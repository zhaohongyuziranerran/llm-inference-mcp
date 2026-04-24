"""Token counting and cost estimation utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate token ratios (conservative estimates)
CHARS_PER_TOKEN = {
    "en": 4.0,       # English: ~4 chars per token
    "zh": 1.5,       # Chinese: ~1.5 chars per token
    "code": 3.5,     # Code: ~3.5 chars per token
    "mixed": 2.5,    # Mixed: ~2.5 chars per token
}


def estimate_tokens(text: str, language: str = "mixed") -> int:
    """
    Estimate token count for text.
    Uses character-based heuristics when tiktoken is not available.
    """
    if not text:
        return 0

    # Try to detect language mix
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text)

    if total_chars == 0:
        return 0

    chinese_ratio = chinese_chars / total_chars

    if chinese_ratio > 0.7:
        ratio = CHARS_PER_TOKEN["zh"]
    elif chinese_ratio < 0.1:
        ratio = CHARS_PER_TOKEN["en"]
    else:
        ratio = CHARS_PER_TOKEN["mixed"]

    return max(1, int(total_chars / ratio))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        # Message overhead (~4 tokens per message for role, separators, etc.)
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Multi-modal content
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total += estimate_tokens(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        # Image tokens depend on resolution, estimate low/med/high
                        detail = part.get("image_url", {}).get("detail", "auto")
                        total += {"low": 85, "high": 1105, "auto": 550}.get(detail, 550)

    return total


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    """Calculate estimated cost in USD."""
    return (input_tokens * input_price_per_1m / 1_000_000 +
            output_tokens * output_price_per_1m / 1_000_000)


def format_cost(cost_usd: float) -> str:
    """Format cost for display."""
    if cost_usd < 0.001:
        return f"${cost_usd*1_000_000:.1f}μ"
    elif cost_usd < 0.01:
        return f"${cost_usd*1000:.2f}m"
    elif cost_usd < 1.0:
        return f"${cost_usd:.4f}"
    else:
        return f"${cost_usd:.2f}"
