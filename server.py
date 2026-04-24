"""LLM Inference MCP Server - Root entry point for MCPize compatibility."""

from llm_inference_mcp.server import mcp

if __name__ == "__main__":
    mcp.run()
