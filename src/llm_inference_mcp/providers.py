"""LLM Provider adapters - Unified interface for multiple LLM backends."""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model specification and pricing."""
    id: str
    name: str
    provider: str
    context_window: int
    max_output: int
    input_price_per_1m: float  # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    supports_structured: bool = False
    supports_vision: bool = False
    supports_tools: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result from an LLM inference call."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw_response: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.client = httpx.Client(timeout=120.0)

    @abstractmethod
    def get_models(self) -> list[ModelInfo]:
        """Return available models for this provider."""
        ...

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
    ) -> InferenceResult:
        """Perform a chat completion."""
        ...

    async def close(self):
        self.client.close()


class OpenAICompatibleProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, DeepSeek, vLLM, etc.)."""

    PROVIDER_MODELS: dict[str, list[ModelInfo]] = {
        "openai": [
            ModelInfo("gpt-4o", "GPT-4o", "openai", 128000, 16384, 2.5, 10.0, True, True, True, ["flagship", "vision"]),
            ModelInfo("gpt-4o-mini", "GPT-4o Mini", "openai", 128000, 16384, 0.15, 0.6, True, True, True, ["fast", "cheap"]),
            ModelInfo("gpt-4-turbo", "GPT-4 Turbo", "openai", 128000, 4096, 10.0, 30.0, False, True, True, ["legacy"]),
            ModelInfo("o3-mini", "O3 Mini", "openai", 200000, 100000, 1.1, 4.4, False, False, False, ["reasoning"]),
        ],
        "deepseek": [
            ModelInfo("deepseek-chat", "DeepSeek V3", "deepseek", 64000, 8192, 0.14, 0.28, False, False, True, ["cheap", "fast"]),
            ModelInfo("deepseek-reasoner", "DeepSeek R1", "deepseek", 64000, 8192, 0.55, 2.19, False, False, False, ["reasoning"]),
        ],
        "vllm": [
            ModelInfo("qwen2.5-7b-instruct", "Qwen2.5-7B (vLLM)", "vllm", 32768, 8192, 0.0, 0.0, False, False, True, ["local", "free"]),
        ],
    }

    def __init__(self, provider_type: str = "openai", **kwargs):
        super().__init__(**kwargs)
        self.provider_type = provider_type
        if not self.base_url:
            default_urls = {
                "openai": "https://api.openai.com/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "vllm": "http://localhost:8000/v1",
            }
            self.base_url = default_urls.get(provider_type, "https://api.openai.com/v1")

    def get_models(self) -> list[ModelInfo]:
        return self.PROVIDER_MODELS.get(self.provider_type, [])

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
    ) -> InferenceResult:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools

        start = time.time()
        resp = await self.client.post(url, json=payload, headers=headers)
        latency = (time.time() - start) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        usage = data.get("usage", {})

        model_info = self._find_model(model)
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = self._calculate_cost(model_info, input_tokens, output_tokens)

        return InferenceResult(
            content=content,
            model=data.get("model", model),
            provider=self.provider_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency,
            finish_reason=choice.get("finish_reason", ""),
            raw_response=data,
        )

    def _find_model(self, model_id: str) -> ModelInfo:
        for m in self.get_models():
            if m.id == model_id:
                return m
        return ModelInfo(model_id, model_id, self.provider_type, 4096, 2048, 0.0, 0.0)

    @staticmethod
    def _calculate_cost(model: ModelInfo, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * model.input_price_per_1m / 1_000_000 +
                output_tokens * model.output_price_per_1m / 1_000_000)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    MODELS = [
        ModelInfo("claude-sonnet-4-20250514", "Claude Sonnet 4", "anthropic", 200000, 64000, 3.0, 15.0, True, True, True, ["flagship", "reasoning"]),
        ModelInfo("claude-haiku-3-5-20241022", "Claude 3.5 Haiku", "anthropic", 200000, 8192, 0.8, 4.0, True, True, True, ["fast", "cheap"]),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.base_url:
            self.base_url = "https://api.anthropic.com"

    def get_models(self) -> list[ModelInfo]:
        return self.MODELS

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
    ) -> InferenceResult:
        url = f"{self.base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI-format messages to Anthropic format
        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n"
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            payload["system"] = system_msg.strip()

        start = time.time()
        resp = await self.client.post(url, json=payload, headers=headers)
        latency = (time.time() - start) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        input_tokens = data.get("usage", {}).get("input_tokens", 0)
        output_tokens = data.get("usage", {}).get("output_tokens", 0)

        model_info = self._find_model(model)
        cost = OpenAICompatibleProvider._calculate_cost(model_info, input_tokens, output_tokens)

        return InferenceResult(
            content=content,
            model=data.get("model", model),
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency,
            finish_reason=data.get("stop_reason", ""),
            raw_response=data,
        )

    def _find_model(self, model_id: str) -> ModelInfo:
        for m in self.MODELS:
            if m.id == model_id:
                return m
        return ModelInfo(model_id, model_id, "anthropic", 200000, 8192, 3.0, 15.0)


class ProviderRegistry:
    """Registry of all configured LLM providers."""

    def __init__(self):
        self.providers: dict[str, BaseProvider] = {}
        self._setup_defaults()

    def _setup_defaults(self):
        """Setup providers from environment variables."""
        # OpenAI
        if api_key := os.environ.get("OPENAI_API_KEY"):
            self.providers["openai"] = OpenAICompatibleProvider(
                provider_type="openai",
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", ""),
            )

        # DeepSeek
        if api_key := os.environ.get("DEEPSEEK_API_KEY"):
            self.providers["deepseek"] = OpenAICompatibleProvider(
                provider_type="deepseek",
                api_key=api_key,
                base_url=os.environ.get("DEEPSEEK_BASE_URL", ""),
            )

        # vLLM (local, no API key needed)
        vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.providers["vllm"] = OpenAICompatibleProvider(
            provider_type="vllm",
            api_key="",
            base_url=vllm_url,
        )

        # Anthropic
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            self.providers["anthropic"] = AnthropicProvider(
                api_key=api_key,
                base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
            )

        # Custom OpenAI-compatible providers
        for i in range(1, 6):
            prefix = f"CUSTOM_LLM_{i}"
            if api_key := os.environ.get(f"{prefix}_API_KEY"):
                self.providers[f"custom_{i}"] = OpenAICompatibleProvider(
                    provider_type="openai",
                    api_key=api_key,
                    base_url=os.environ.get(f"{prefix}_BASE_URL", ""),
                )

    def get_all_models(self) -> list[ModelInfo]:
        """Get all available models across providers."""
        models = []
        for provider in self.providers.values():
            models.extend(provider.get_models())
        return models

    def get_provider(self, provider_name: str) -> BaseProvider | None:
        return self.providers.get(provider_name)

    def find_model(self, model_id: str) -> tuple[BaseProvider | None, ModelInfo | None]:
        """Find provider and model info for a model ID."""
        for provider in self.providers.values():
            for m in provider.get_models():
                if m.id == model_id:
                    return provider, m
        return None, None

    def find_cheapest_model(self, tags: list[str] | None = None) -> tuple[BaseProvider | None, ModelInfo | None]:
        """Find the cheapest available model, optionally filtered by tags."""
        candidates = []
        for provider in self.providers.values():
            for m in provider.get_models():
                if tags and not any(t in m.tags for t in tags):
                    continue
                candidates.append((provider, m))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[1].output_price_per_1m)
        return candidates[0]

    def find_fastest_model(self, tags: list[str] | None = None) -> tuple[BaseProvider | None, ModelInfo | None]:
        """Find the fastest (cheapest input) model for quick tasks."""
        candidates = []
        for provider in self.providers.values():
            for m in provider.get_models():
                if tags and not any(t in m.tags for t in tags):
                    continue
                candidates.append((provider, m))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[1].input_price_per_1m)
        return candidates[0]
