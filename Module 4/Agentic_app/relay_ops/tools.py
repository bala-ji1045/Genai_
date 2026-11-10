"""Local agent tools (Lab 1, Section A): plain Python functions the agent calls in-process.

| Tool                  | Type                      | Risk | Policy                       |
|-----------------------|---------------------------|------|------------------------------|
| lookup_fact           | Retrieval — list KB       | Low  | runs automatically           |
| search_knowledge_base | Retrieval — RAG           | Low  | runs automatically           |
| run_python            | Computation (PoT)         | Low  | runs automatically (sandbox) |
| send_report           | Action — sends an email   | HIGH | human approval (interrupt)   |

The MCP server (mcp_server/knowledge_server.py) exposes the same retrieval tools under the same names,
so the agents can use either source without any other change.
"""
import math
from datetime import datetime

from langchain_core.tools import tool

from . import knowledge, rag


@tool
def lookup_fact(query: str) -> str:
    """Look up facts about a named place or product in the course knowledge base,
    e.g. 'mount everlore'. Use for factual questions. Does NOT do arithmetic."""
    return knowledge.lookup(query)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the Relay Ops document library for passages that answer a question,
    e.g. 'how long does the communicator battery last'. Returns passages with their source.
    Use for factual questions. Does NOT do arithmetic."""
    return rag.search_text(query)


SAFE_BUILTINS = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum,
                 "int": int, "float": float, "len": len, "range": range, "pow": pow}


@tool
def run_python(code: str) -> str:
    """Program-of-thought calculator: runs a short Python snippet and returns the result.
    Write plain Python such as: answer = 847 * 23 + 156
    Use this for ANY arithmetic instead of computing it yourself."""
    code = code.strip().strip("`").removeprefix("python").strip()          # tolerate ```python fences
    if any(bad in code for bad in ("import", "open(", "__", "exec", "eval")):
        return "Error: imports, files and dunder access are not allowed in the sandbox."
    lines = code.splitlines() or ["None"]
    if "answer" not in code:                                               # REPL-style: a bare expression
        lines[-1] = f"answer = {lines[-1]}"                                #   becomes the answer
    scope = {"__builtins__": SAFE_BUILTINS, "math": math}
    try:
        exec("\n".join(lines), scope)
    except Exception as e:
        return f"Error while running the code: {e}"          # errors become observations, not crashes
    ans = scope.get("answer")
    return str(round(ans, 4) if isinstance(ans, float) else ans)


OUTBOX = []          # stands in for a real mail server; the UI shows it on the "Outbox" tab


@tool
def send_report(to: str, subject: str, body: str) -> str:
    """Email a report. HIGH RISK: cannot be undone. Only call after a human has approved it."""
    OUTBOX.append({"time": datetime.now().strftime("%H:%M:%S"), "to": to, "subject": subject, "body": body})
    return f"Email sent to {to}."
