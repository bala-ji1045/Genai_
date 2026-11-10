"""MCP client side (Lab 1, Section F): start the knowledge server if needed, then discover its tools.

The server runs as its own long-lived process over streamable HTTP, so the RAG models are loaded once
and every tool call is just an HTTP request — the way an MCP server is deployed in practice.
"""
import atexit
import socket
import subprocess
import sys
import time

from langchain_mcp_adapters.client import MultiServerMCPClient

from . import config

_server_process = None


def server_is_up() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((config.MCP_HOST, config.MCP_PORT)) == 0


def ensure_server(timeout: float = 180) -> str:
    """Start mcp_server/knowledge_server.py in the background unless something already listens on the port."""
    global _server_process
    if server_is_up():
        return "already running"
    if _server_process is None or _server_process.poll() is not None:
        log = open(config.APP_DIR / "mcp_server.log", "w")
        _server_process = subprocess.Popen([sys.executable, str(config.MCP_SERVER_SCRIPT)],
                                           stdout=log, stderr=subprocess.STDOUT, cwd=config.APP_DIR)
        atexit.register(_server_process.terminate)        # a server we started stops when we stop
    deadline = time.time() + timeout                       # first start downloads/loads the RAG models
    while time.time() < deadline:
        if server_is_up():
            return "started"
        if _server_process.poll() is not None:
            raise RuntimeError("The MCP server exited during start-up. See mcp_server.log in the app folder.")
        time.sleep(0.5)
    raise RuntimeError(f"The MCP server did not open {config.MCP_URL} within {timeout:.0f}s. See mcp_server.log.")


def get_client() -> MultiServerMCPClient:
    return MultiServerMCPClient({"knowledge": {"url": config.MCP_URL, "transport": "streamable_http"}})


async def get_mcp_tools() -> dict:
    """tools/list over MCP -> {name: LangChain tool}. Nothing about these tools is hard-coded in the app."""
    ensure_server()
    return {t.name: t for t in await get_client().get_tools()}
