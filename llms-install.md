# LLM Inference MCP - Installation Guide

## Quick Start

Add this to your MCP client configuration (Claude Desktop, Cursor, Windsurf, etc.):

```json
{
  "mcpServers": {
    "llm-inference": {
      "command": "uvx",
      "args": ["llm-inference-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "DEEPSEEK_API_KEY": "sk-...",
        "VLLM_BASE_URL": "http://localhost:8000/v1"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | OpenAI API key |
| `OPENAI_BASE_URL` | No | Custom OpenAI endpoint |
| `DEEPSEEK_API_KEY` | No | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | No | Custom DeepSeek endpoint |
| `ANTHROPIC_API_KEY` | No | Anthropic API key |
| `ANTHROPIC_BASE_URL` | No | Custom Anthropic endpoint |
| `VLLM_BASE_URL` | No | Local vLLM endpoint (default: http://localhost:8000/v1) |
| `CUSTOM_LLM_1_API_KEY` | No | Custom provider API key |
| `CUSTOM_LLM_1_BASE_URL` | No | Custom provider endpoint |

At least one provider must be configured. vLLM is enabled by default (no API key needed).
