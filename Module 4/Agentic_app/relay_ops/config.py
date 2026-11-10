"""All settings in one place. Each can be overridden with an environment variable."""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("RELAY_DOCS_DIR", APP_DIR / "data" / "docs"))

# --- LLM (Ollama) -----------------------------------------------------------------------------
MODEL = os.getenv("RELAY_MODEL", "qwen3:1.7b")          # "qwen3:4b" / "qwen3:8b" if you have the RAM
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- RAG (same models as the Module 3 lab) ----------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3

# --- MCP server -------------------------------------------------------------------------------
MCP_HOST = os.getenv("RELAY_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("RELAY_MCP_PORT", "8765"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"
MCP_SERVER_SCRIPT = APP_DIR / "mcp_server" / "knowledge_server.py"

# --- Harness ----------------------------------------------------------------------------------
RECURSION_LIMIT = 25
RISKY_TOOLS = {"send_report"}          # the harness asks a human before running any of these
