"""LLM Inference MCP Server - Multi-model routing, cost optimization, structured output.

Provides 7 tools for intelligent LLM inference management:
1. chat_completion - Smart routing chat completion
2. structured_output - Force JSON/schema output
3. batch_inference - Process multiple prompts in parallel
4. count_tokens - Token counting for any model
5. estimate_cost - Cost estimation before running
6. list_models - List available models and specs
7. compare_models - Run same prompt on multiple models
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

from fastmcp import FastMCP

from .providers import ProviderRegistry, InferenceResult
from .router import SmartRouter
from .tokenizer import estimate_tokens, estimate_messages_tokens, estimate_cost, format_cost

logger = logging.getLogger(__name__)

# Initialize FastMCP 3.x server
mcp = FastMCP(
    "llm-inference-mcp",
    instructions=(
        "LLM Inference MCP Server - Multi-model routing with cost optimization. "
        "Supports OpenAI, DeepSeek, Anthropic Claude, and local vLLM backends. "
        "Features: smart model routing, structured output enforcement, batch inference, "
        "token counting, cost estimation, and model comparison. "
        "Set environment variables to configure providers: "
        "OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, VLLM_BASE_URL."
    ),
)

# Global state
_registry = ProviderRegistry()
_router = SmartRouter(_registry)
_usage_stats: dict[str, dict] = {}  # Track usage per model


@mcp.tool()
async def list_models(
    provider: str = "",
    tag: str = "",
) -> str:
    """List all available LLM models with specifications and pricing.

    Args:
        provider: Filter by provider name (openai/deepseek/anthropic/vllm). Empty = all.
        tag: Filter by tag (fast/cheap/flagship/reasoning/local/free). Empty = all.
    """
    models = _registry.get_all_models()

    if provider:
        models = [m for m in models if m.provider == provider.lower()]
    if tag:
        models = [m for m in models if tag.lower() in m.tags]

    if not models:
        return f"No models found. Provider={provider or 'all'}, Tag={tag or 'all'}"

    lines = ["# Available LLM Models\n"]
    lines.append("| # | Model | Provider | Context | Max Out | Input $/1M | Output $/1M | Structured | Vision | Tags |")
    lines.append("|---|-------|----------|---------|---------|------------|-------------|------------|--------|------|")

    for i, m in enumerate(models, 1):
        struct_icon = "Y" if m.supports_structured else "-"
        vision_icon = "Y" if m.supports_vision else "-"
        input_price = f"${m.input_price_per_1m}" if m.input_price_per_1m > 0 else "FREE"
        output_price = f"${m.output_price_per_1m}" if m.output_price_per_1m > 0 else "FREE"
        tags_str = ", ".join(m.tags)
        lines.append(
            f"| {i} | {m.name} | {m.provider} | {m.context_window:,} | {m.max_output:,} | "
            f"{input_price} | {output_price} | {struct_icon} | {vision_icon} | {tags_str} |"
        )

    lines.append(f"\n**Total: {len(models)} models from {len(set(m.provider for m in models))} providers**")
    lines.append("\nConfigure providers via environment variables: OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, VLLM_BASE_URL")

    return "\n".join(lines)


@mcp.tool()
async def chat_completion(
    messages: list[dict],
    model: str = "",
    provider: str = "",
    task_type: str = "chat",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    prefer_cheapest: bool = False,
) -> str:
    """Send a chat completion request with intelligent model routing.

    If no model is specified, automatically selects the best model based on task_type.
    Supports fallback to alternative models if the primary fails.

    Args:
        messages: Chat messages in OpenAI format [{"role": "user", "content": "..."}]
        model: Specific model ID (empty = auto-select via smart routing)
        provider: Preferred provider (empty = auto-select)
        task_type: Task type for smart routing (quick/reasoning/creative/code/extraction/chat/translation/summary)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum output tokens
        prefer_cheapest: If true, prioritize cheapest model
    """
    if not messages:
        return "Error: messages cannot be empty"

    # Smart routing
    selected_provider, selected_model, reason = _router.route(
        task_type=task_type,
        prefer_model=model if model else None,
        prefer_provider=provider if provider else None,
        require_structured=False,
    )

    if prefer_cheapest and not model:
        cheapest_provider, cheapest_model = _registry.find_cheapest_model()
        if cheapest_provider and cheapest_model:
            selected_provider = cheapest_provider
            selected_model = cheapest_model
            reason = f"Cheapest model: {cheapest_model.name}"

    if not selected_provider or not selected_model:
        available = [m.id for m in _registry.get_all_models()]
        return f"Error: No model available. Configured providers: {list(_registry.providers.keys())}. Available models: {available}"

    # Try primary model
    try:
        result = await selected_provider.chat_completion(
            model=selected_model.id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        _track_usage(result)
        return _format_result(result, reason)

    except Exception as e:
        logger.warning(f"Primary model failed: {e}")

    # Fallback: try other providers
    for pname, prov in _registry.providers.items():
        if prov == selected_provider:
            continue
        models = prov.get_models()
        if not models:
            continue
        try:
            fallback_model = models[0]
            result = await prov.chat_completion(
                model=fallback_model.id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _track_usage(result)
            return _format_result(result, f"FALLBACK from {selected_model.name}: {str(e)[:100]}")

        except Exception as e2:
            logger.warning(f"Fallback {pname} also failed: {e2}")
            continue

    return f"Error: All providers failed. Primary error: {str(e)[:200]}"


@mcp.tool()
async def structured_output(
    messages: list[dict],
    schema: dict,
    model: str = "",
    provider: str = "",
    task_type: str = "extraction",
    max_tokens: int = 4096,
) -> str:
    """Force structured JSON output from any LLM model matching a JSON schema.

    For models with native structured output support (GPT-4o, etc.), uses response_format.
    For others, injects schema instructions into the prompt and parses the JSON response.

    Args:
        messages: Chat messages in OpenAI format
        schema: JSON Schema that the output must conform to (e.g., {"type": "object", "properties": {...}})
        model: Specific model ID (empty = auto-select)
        provider: Preferred provider (empty = auto-select)
        task_type: Task type for routing (default: extraction)
        max_tokens: Maximum output tokens
    """
    if not messages or not schema:
        return "Error: messages and schema are required"

    # Try to find a model with native structured output support
    selected_provider, selected_model, reason = _router.route(
        task_type=task_type,
        prefer_model=model if model else None,
        prefer_provider=provider if provider else None,
        require_structured=True,
    )

    # If no structured-capable model, use prompt engineering
    use_native_structured = selected_model and selected_model.supports_structured

    if use_native_structured and selected_provider:
        try:
            result = await selected_provider.chat_completion(
                model=selected_model.id,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": schema}},
            )
            _track_usage(result)
            return _format_structured_result(result, reason, native=True)

        except Exception as e:
            logger.warning(f"Native structured output failed: {e}, falling back to prompt engineering")

    # Fallback: prompt engineering for any model
    schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
    structured_prompt = (
        f"You MUST respond with valid JSON that conforms to this schema:\n"
        f"```json\n{schema_str}\n```\n\n"
        f"Important rules:\n"
        f"1. Output ONLY valid JSON, no markdown, no explanation\n"
        f"2. All required fields must be present\n"
        f"3. Use proper types (string, number, boolean, array, object)\n"
        f"4. If you cannot determine a value, use null\n"
    )

    # Inject schema instruction
    enhanced_messages = list(messages)
    last_msg = enhanced_messages[-1]
    if last_msg["role"] == "user":
        enhanced_messages[-1] = {
            "role": "user",
            "content": last_msg["content"] + "\n\n" + structured_prompt,
        }
    else:
        enhanced_messages.append({"role": "user", "content": structured_prompt})

    # Re-route without structured requirement
    if not selected_provider:
        selected_provider, selected_model, reason = _router.route(
            task_type=task_type,
            prefer_model=model if model else None,
        )

    if not selected_provider or not selected_model:
        return "Error: No model available for structured output"

    try:
        result = await selected_provider.chat_completion(
            model=selected_model.id,
            messages=enhanced_messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        _track_usage(result)

        # Try to parse and validate JSON
        content = result.content.strip()
        if content.startswith("```"):
            # Strip markdown code blocks
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if len(lines) > 2 else lines)

        try:
            parsed = json.loads(content)
            result.content = json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass  # Return as-is, let user handle

        return _format_structured_result(result, f"{reason} (prompt-engineered)", native=False)

    except Exception as e:
        return f"Error: Structured output failed: {str(e)[:300]}"


@mcp.tool()
async def batch_inference(
    prompts: list[str],
    system_prompt: str = "",
    model: str = "",
    provider: str = "",
    task_type: str = "quick",
    temperature: float = 0.5,
    max_tokens: int = 1024,
) -> str:
    """Process multiple prompts in parallel using the same model settings.

    Efficiently handles batch requests by running them concurrently.
    Returns all results with aggregated token counts and costs.

    Args:
        prompts: List of prompt strings to process
        system_prompt: Optional system prompt applied to all requests
        model: Specific model ID (empty = auto-select cheapest for task)
        provider: Preferred provider (empty = auto-select)
        task_type: Task type for routing (default: quick for batch)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens per prompt
    """
    if not prompts:
        return "Error: prompts list cannot be empty"

    if len(prompts) > 50:
        return "Error: Maximum 50 prompts per batch request"

    # Auto-select model (prefer cheapest for batch)
    selected_provider, selected_model, reason = _router.route(
        task_type=task_type,
        prefer_model=model if model else None,
        prefer_provider=provider if provider else None,
    )

    if not selected_provider or not selected_model:
        return "Error: No model available for batch inference"

    # Build messages for each prompt
    async def process_one(prompt: str, index: int) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            result = await selected_provider.chat_completion(
                model=selected_model.id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _track_usage(result)
            return {
                "index": index,
                "status": "success",
                "content": result.content,
                "tokens": result.input_tokens + result.output_tokens,
                "cost": result.cost_usd,
            }
        except Exception as e:
            return {
                "index": index,
                "status": "error",
                "content": str(e)[:200],
                "tokens": 0,
                "cost": 0.0,
            }

    # Run all in parallel
    start = time.time()
    tasks = [process_one(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    total_time = (time.time() - start) * 1000

    # Aggregate
    success_count = sum(1 for r in results if r["status"] == "success")
    total_tokens = sum(r["tokens"] for r in results)
    total_cost = sum(r["cost"] for r in results)

    lines = [f"# Batch Inference Results\n"]
    lines.append(f"**Model**: {selected_model.name} | **Prompt**: {reason}")
    lines.append(f"**Success**: {success_count}/{len(prompts)} | **Time**: {total_time:.0f}ms | **Tokens**: {total_tokens:,} | **Cost**: {format_cost(total_cost)}\n")

    for r in sorted(results, key=lambda x: x["index"]):
        status = "OK" if r["status"] == "success" else "ERR"
        lines.append(f"## [{status}] Prompt #{r['index'] + 1}")
        lines.append(f"{r['content']}\n")

    return "\n".join(lines)


@mcp.tool()
async def count_tokens(
    text: str = "",
    messages: list[dict] | None = None,
) -> str:
    """Count estimated tokens for text or chat messages.

    Useful for estimating costs before making API calls and ensuring inputs
    fit within model context windows.

    Args:
        text: Plain text to count tokens for
        messages: Chat messages in OpenAI format (alternative to text)
    """
    if text:
        tokens = estimate_tokens(text)
        # Estimate for different models
        model_fits = []
        for m in _registry.get_all_models():
            fits = "YES" if tokens <= m.context_window else "NO"
            model_fits.append(f"  {m.name} ({m.context_window:,}): {fits}")

        return (
            f"# Token Count Estimate\n"
            f"**Text length**: {len(text)} chars\n"
            f"**Estimated tokens**: ~{tokens:,}\n\n"
            f"## Model Compatibility\n" + "\n".join(model_fits)
        )

    elif messages:
        tokens = estimate_messages_tokens(messages)
        return (
            f"# Token Count Estimate\n"
            f"**Messages**: {len(messages)}\n"
            f"**Estimated tokens**: ~{tokens:,} (includes message overhead)"
        )

    else:
        return "Error: Provide either text or messages parameter"


@mcp.tool()
async def estimate_inference_cost(
    input_text: str = "",
    output_tokens: int = 500,
    model: str = "",
) -> str:
    """Estimate the cost of an inference call before making it.

    Helps optimize spending by comparing costs across models.

    Args:
        input_text: The input text/prompt
        output_tokens: Expected number of output tokens
        model: Specific model to estimate for (empty = compare all)
    """
    input_tokens = estimate_tokens(input_text) if input_text else 0
    models = _registry.get_all_models()

    if model:
        models = [m for m in models if m.id == model or m.name == model]

    if not models:
        return f"No models found for: {model or 'all'}"

    lines = [f"# Cost Estimation\n"]
    lines.append(f"**Input tokens**: ~{input_tokens:,} | **Output tokens**: ~{output_tokens:,}\n")
    lines.append("| Model | Provider | Input Cost | Output Cost | Total Cost |")
    lines.append("|-------|----------|------------|-------------|------------|")

    for m in sorted(models, key=lambda x: estimate_cost(input_tokens, output_tokens, x.input_price_per_1m, x.output_price_per_1m)):
        ic = estimate_cost(input_tokens, 0, m.input_price_per_1m, 0)
        oc = estimate_cost(0, output_tokens, 0, m.output_price_per_1m)
        tc = estimate_cost(input_tokens, output_tokens, m.input_price_per_1m, m.output_price_per_1m)
        lines.append(f"| {m.name} | {m.provider} | {format_cost(ic)} | {format_cost(oc)} | **{format_cost(tc)}** |")

    return "\n".join(lines)


@mcp.tool()
async def compare_models(
    messages: list[dict],
    models: list[str] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Run the same prompt on multiple models and compare results side by side.

    Useful for evaluating model quality, latency, and cost for specific tasks.

    Args:
        messages: Chat messages in OpenAI format
        models: List of model IDs to compare (empty = compare all available)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
    """
    if not messages:
        return "Error: messages cannot be empty"

    all_models = _registry.get_all_models()
    if models:
        all_models = [m for m in all_models if m.id in models]

    if not all_models:
        return "No models available for comparison"

    if len(all_models) > 5:
        all_models = all_models[:5]

    # Run inference on each model in parallel
    async def run_one(model_info):
        provider, _ = _registry.find_model(model_info.id)
        if not provider:
            return {"model": model_info.name, "status": "error", "content": "Provider not configured"}

        try:
            result = await provider.chat_completion(
                model=model_info.id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _track_usage(result)
            return {
                "model": model_info.name,
                "provider": model_info.provider,
                "status": "success",
                "content": result.content,
                "tokens": result.input_tokens + result.output_tokens,
                "cost": result.cost_usd,
                "latency": result.latency_ms,
            }
        except Exception as e:
            return {
                "model": model_info.name,
                "provider": model_info.provider,
                "status": "error",
                "content": str(e)[:200],
                "tokens": 0,
                "cost": 0.0,
                "latency": 0,
            }

    results = await asyncio.gather(*[run_one(m) for m in all_models])

    # Format comparison
    lines = ["# Model Comparison\n"]
    lines.append("| Model | Provider | Status | Tokens | Cost | Latency |")
    lines.append("|-------|----------|--------|--------|------|---------|")

    for r in sorted(results, key=lambda x: x.get("latency", 999999)):
        status = "OK" if r["status"] == "success" else "ERR"
        tokens = f"{r['tokens']:,}" if r["tokens"] else "-"
        cost = format_cost(r["cost"]) if r["cost"] else "-"
        latency = f"{r['latency']:.0f}ms" if r["latency"] else "-"
        lines.append(f"| {r['model']} | {r['provider']} | {status} | {tokens} | {cost} | {latency} |")

    lines.append("\n---\n")
    for r in results:
        lines.append(f"## {r['model']} ({r['provider']})")
        lines.append(f"{r['content']}\n")

    return "\n".join(lines)


def _track_usage(result: InferenceResult):
    """Track usage statistics per model."""
    key = f"{result.provider}/{result.model}"
    if key not in _usage_stats:
        _usage_stats[key] = {"calls": 0, "tokens": 0, "cost": 0.0}
    _usage_stats[key]["calls"] += 1
    _usage_stats[key]["tokens"] += result.input_tokens + result.output_tokens
    _usage_stats[key]["cost"] += result.cost_usd


def _format_result(result: InferenceResult, reason: str) -> str:
    """Format an inference result for display."""
    lines = [
        f"# Inference Result\n",
        f"**Model**: {result.model} ({result.provider}) | **Routing**: {reason}",
        f"**Tokens**: {result.input_tokens:,} in + {result.output_tokens:,} out | **Cost**: {format_cost(result.cost_usd)} | **Latency**: {result.latency_ms:.0f}ms",
        f"\n---\n\n{result.content}",
    ]
    return "\n".join(lines)


def _format_structured_result(result: InferenceResult, reason: str, native: bool) -> str:
    """Format a structured output result."""
    method = "native structured output" if native else "prompt engineering"
    lines = [
        f"# Structured Output\n",
        f"**Model**: {result.model} ({result.provider}) | **Method**: {method}",
        f"**Tokens**: {result.input_tokens:,} in + {result.output_tokens:,} out | **Cost**: {format_cost(result.cost_usd)}",
        f"\n---\n\n```json\n{result.content}\n```",
    ]
    return "\n".join(lines)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
