"""Smart Router - Intelligent model selection based on task requirements."""

import logging
from typing import Optional

from .providers import ProviderRegistry, ModelInfo, BaseProvider

logger = logging.getLogger(__name__)


class SmartRouter:
    """Routes inference requests to the optimal model based on task requirements."""

    # Task type to model tag mapping
    TASK_PROFILES = {
        "quick": {
            "description": "Quick, simple tasks - prioritize speed and cost",
            "preferred_tags": ["fast", "cheap"],
            "avoid_tags": ["reasoning", "flagship"],
            "max_temperature": 1.0,
        },
        "reasoning": {
            "description": "Complex reasoning and analysis - prioritize quality",
            "preferred_tags": ["reasoning", "flagship"],
            "avoid_tags": ["cheap"],
            "max_temperature": 0.3,
        },
        "creative": {
            "description": "Creative writing and brainstorming - balance quality and diversity",
            "preferred_tags": ["flagship"],
            "avoid_tags": [],
            "max_temperature": 1.2,
        },
        "code": {
            "description": "Code generation and debugging - prioritize accuracy",
            "preferred_tags": ["flagship", "reasoning"],
            "avoid_tags": ["cheap"],
            "max_temperature": 0.2,
        },
        "extraction": {
            "description": "Data extraction and structured output - prioritize reliability",
            "preferred_tags": ["fast"],
            "avoid_tags": [],
            "max_temperature": 0.0,
        },
        "chat": {
            "description": "General chat and Q&A - balance speed and quality",
            "preferred_tags": ["fast", "cheap"],
            "avoid_tags": ["reasoning"],
            "max_temperature": 0.8,
        },
        "translation": {
            "description": "Translation tasks - prioritize accuracy and fluency",
            "preferred_tags": ["flagship"],
            "avoid_tags": [],
            "max_temperature": 0.3,
        },
        "summary": {
            "description": "Summarization tasks - balance speed and quality",
            "preferred_tags": ["fast", "cheap"],
            "avoid_tags": ["reasoning"],
            "max_temperature": 0.5,
        },
    }

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    def route(
        self,
        task_type: str = "chat",
        prefer_model: str | None = None,
        prefer_provider: str | None = None,
        max_cost_per_1m: float | None = None,
        require_vision: bool = False,
        require_structured: bool = False,
        require_tools: bool = False,
    ) -> tuple[BaseProvider | None, ModelInfo | None, str]:
        """
        Select the best model for a task.
        Returns (provider, model_info, reason).
        """
        # 1. If specific model requested, try it
        if prefer_model:
            provider, model = self.registry.find_model(prefer_model)
            if provider and model:
                return provider, model, f"User-specified model: {model.name}"

        # 2. If specific provider requested, pick best model from that provider
        if prefer_provider:
            provider = self.registry.get_provider(prefer_provider)
            if provider:
                models = provider.get_models()
                if models:
                    model = self._select_best(models, task_type, max_cost_per_1m,
                                              require_vision, require_structured, require_tools)
                    if model:
                        return provider, model, f"Best from {prefer_provider}: {model.name}"

        # 3. Smart routing based on task type
        profile = self.TASK_PROFILES.get(task_type, self.TASK_PROFILES["chat"])
        candidates = []

        for provider in self.registry.providers.values():
            for m in provider.get_models():
                # Filter by requirements
                if require_vision and not m.supports_vision:
                    continue
                if require_structured and not m.supports_structured:
                    continue
                if require_tools and not m.supports_tools:
                    continue
                if max_cost_per_1m and m.output_price_per_1m > max_cost_per_1m:
                    continue

                # Score the model
                score = self._score_model(m, profile)
                candidates.append((provider, m, score))

        if not candidates:
            return None, None, "No suitable model found"

        candidates.sort(key=lambda x: x[2], reverse=True)
        best_provider, best_model, best_score = candidates[0]
        reason = f"Smart route for '{task_type}': {best_model.name} (score: {best_score:.1f})"

        return best_provider, best_model, reason

    def _score_model(self, model: ModelInfo, profile: dict) -> float:
        """Score a model for a given task profile (higher = better)."""
        score = 50.0  # Base score

        preferred = profile.get("preferred_tags", [])
        avoid = profile.get("avoid_tags", [])

        for tag in preferred:
            if tag in model.tags:
                score += 20.0

        for tag in avoid:
            if tag in model.tags:
                score -= 15.0

        # Cost bonus (cheaper = better)
        if model.output_price_per_1m == 0:
            score += 25.0  # Free!
        elif model.output_price_per_1m < 1.0:
            score += 15.0
        elif model.output_price_per_1m < 5.0:
            score += 5.0
        elif model.output_price_per_1m > 20.0:
            score -= 10.0

        # Context window bonus
        if model.context_window >= 128000:
            score += 5.0

        # Structured output bonus (always nice to have)
        if model.supports_structured:
            score += 3.0

        return score

    def _select_best(
        self,
        models: list[ModelInfo],
        task_type: str,
        max_cost_per_1m: float | None,
        require_vision: bool,
        require_structured: bool,
        require_tools: bool,
    ) -> ModelInfo | None:
        """Select best model from a list."""
        profile = self.TASK_PROFILES.get(task_type, self.TASK_PROFILES["chat"])
        candidates = []
        for m in models:
            if require_vision and not m.supports_vision:
                continue
            if require_structured and not m.supports_structured:
                continue
            if require_tools and not m.supports_tools:
                continue
            if max_cost_per_1m and m.output_price_per_1m > max_cost_per_1m:
                continue
            candidates.append((m, self._score_model(m, profile)))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
