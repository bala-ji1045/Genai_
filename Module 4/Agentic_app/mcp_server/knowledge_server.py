"""Relay Ops knowledge MCP server.

Exposes the same capabilities as the local agent tools, but from a separate process that ANY MCP client
(this app, Claude Desktop, an IDE, ...) can discover and call.

Tools     : lookup_fact (list KB), search_knowledge_base (RAG), calculate
Resources : kb://{topic}  (one list-KB entry),  docs://list  (the RAG document library)

Run it:
    python mcp_server/knowledge_server.py                     # streamable HTTP on 127.0.0.1:8765/mcp (used by the app)
    python mcp_server/knowledge_server.py --transport stdio   # for Claude Desktop / MCP Inspector
"""
import argparse
import ast
import operator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # so we can import the relay_ops package

from mcp.server.fastmcp import FastMCP                               # noqa: E402

from relay_ops import config, knowledge, rag                         # noqa: E402

mcp = FastMCP("relay-ops-knowledge", host=config.MCP_HOST, port=config.MCP_PORT)


@mcp.tool()
def lookup_fact(query: str) -> str:
    """Look up facts about a named place or product in the course knowledge base,
    e.g. 'mount everlore'. Use for factual questions. Does NOT do arithmetic."""
    return knowledge.lookup(query)


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the Relay Ops document library for passages that answer a question,
    e.g. 'how long does the communicator battery last'. Returns passages with their source.
    Use for factual questions. Does NOT do arithmetic."""
    return rag.search_text(query)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / ** are allowed")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a numeric arithmetic expression, e.g. '340 / 28'."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"(evaluation error: {e})"


@mcp.resource("kb://{topic}")
def kb_resource(topic: str) -> str:
    """Raw list-KB entry for a topic, e.g. kb://mount everlore"""
    return knowledge.lookup(topic)


@mcp.resource("docs://list")
def docs_list() -> str:
    """Names of the documents in the RAG library."""
    return "\n".join(d.metadata["source"] for d in rag.load_documents())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "stdio"], default="http")
    args = parser.parse_args()
    if args.transport == "http":
        rag.get_index()                     # build the RAG index (and load models) once, before the first request
        print(f"Relay Ops MCP server listening on {config.MCP_URL}", file=sys.stderr, flush=True)
    mcp.run(transport="streamable-http" if args.transport == "http" else "stdio")
