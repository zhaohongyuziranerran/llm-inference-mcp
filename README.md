# LLM Inference MCP Server

Multi-model LLM routing with cost optimization, structured output, and batch inference.

## Features

- **Smart Routing**: Automatically selects the best model based on task type (quick/reasoning/creative/code/extraction/chat/translation/summary)
- **Multi-Provider**: Supports OpenAI, DeepSeek, Anthropic Claude, and local vLLM
- **Cost Optimization**: Compare costs across models, track spending, find cheapest options
- **Structured Output**: Force JSON schema output from any model (native or prompt-engineered)
- **Batch Inference**: Process up to 50 prompts in parallel
- **Token Counting**: Estimate tokens for text/messages before making API calls
- **Model Comparison**: Run same prompt on multiple models side-by-side

## Tools

| Tool | Description |
|------|-------------|
| `list_models` | List all available models with specs and pricing |
| `chat_completion` | Smart routing chat completion with auto-fallback |
| `structured_output` | Force JSON schema output from any model |
| `batch_inference` | Process multiple prompts in parallel (up to 50) |
| `count_tokens` | Estimate tokens for text or messages |
| `estimate_inference_cost` | Compare costs across models before running |
| `compare_models` | Run same prompt on multiple models and compare |

## Configuration

Set environment variables to enable providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"  # optional

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_BASE_URL="https://api.anthropic.com"  # optional

# vLLM (local GPU inference)
export VLLM_BASE_URL="http://localhost:8000/v1"  # default

# Custom OpenAI-compatible providers (up to 5)
export CUSTOM_LLM_1_API_KEY="..."
export CUSTOM_LLM_1_BASE_URL="https://..."
```

## Installation

### Using with Claude Desktop / Cursor / Windsurf

Add to your MCP settings:

```json
{
  "mcpServers": {
    "llm-inference": {
      "command": "python",
      "args": ["-m", "llm_inference_mcp.server"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "DEEPSEEK_API_KEY": "sk-...",
        "VLLM_BASE_URL": "http://localhost:8000/v1"
      }
    }
  }
}
```

### Using with uvx

```bash
uvx llm-inference-mcp
```

### Using with pip

```bash
pip install llm-inference-mcp
llm-inference-mcp
```

## Smart Routing

The router automatically selects the best model based on task type:

| Task Type | Description | Prioritizes |
|-----------|-------------|-------------|
| `quick` | Quick, simple tasks | Speed + Cost |
| `reasoning` | Complex analysis | Quality + Reasoning |
| `creative` | Writing, brainstorming | Quality + Diversity |
| `code` | Code generation | Accuracy + Quality |
| `extraction` | Data extraction | Reliability + Speed |
| `chat` | General Q&A | Balance |
| `translation` | Translation | Accuracy |
| `summary` | Summarization | Speed + Cost |

## Cost Optimization Examples

```
# Find cheapest model for a task
chat_completion(messages=[...], prefer_cheapest=True)

# Compare costs before running
estimate_inference_cost(input_text="Analyze this report...", output_tokens=1000)

# Use local vLLM for zero-cost inference
chat_completion(messages=[...], provider="vllm")
```

## Supported Models

| Provider | Models | Input $/1M | Output $/1M |
|----------|--------|------------|-------------|
| OpenAI | GPT-4o, GPT-4o Mini, O3 Mini | $0.15-$10.00 | $0.60-$30.00 |
| DeepSeek | DeepSeek V3, R1 | $0.14-$0.55 | $0.28-$2.19 |
| Anthropic | Claude Sonnet 4, Haiku 3.5 | $0.80-$3.00 | $4.00-$15.00 |
| vLLM | Qwen2.5-7B, any local model | **FREE** | **FREE** |

## License

MIT
