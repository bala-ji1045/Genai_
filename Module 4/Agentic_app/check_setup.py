"""Check that everything the app needs is in place, and say how to fix anything that isn't.

    python check_setup.py           # quick checks (about 1 minute the first time: models download)
    python check_setup.py --full    # also runs the agents end to end on a few questions (local and MCP)
"""
import asyncio
import json
import sys
import urllib.request

OK, FAIL = "  [ OK ]", "  [FAIL]"
problems = []


def check(label, fn, fix):
    try:
        detail = fn()
        print(f"{OK} {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"{FAIL} {label} — {type(e).__name__}: {e}\n         fix: {fix}")
        problems.append(label)
        return False


def python_version():
    assert sys.version_info >= (3, 10), f"found {sys.version.split()[0]}; the mcp package needs 3.10+"
    return sys.version.split()[0]


def packages():
    import importlib.util
    missing = [m for m in ("streamlit", "pandas", "langgraph", "langchain_ollama", "langchain_mcp_adapters", "mcp",
                           "langchain_huggingface", "sentence_transformers", "faiss", "rank_bm25")
               if importlib.util.find_spec(m) is None]
    assert not missing, f"missing: {', '.join(missing)}"


def ollama():
    from relay_ops import config
    try:
        tags = json.load(urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5))
    except OSError:
        raise RuntimeError("Ollama is not running")
    pulled = [m["name"] for m in tags["models"]]
    assert any(n.startswith(config.MODEL) for n in pulled), f"{config.MODEL} not pulled (have: {pulled})"
    return config.MODEL


def llm_replies():
    from relay_ops.llm import get_llm
    return get_llm().invoke("Reply with the single word: ready").content.strip()[:40]


def rag_index():
    from relay_ops import rag
    index = rag.get_index()
    hits = index.search("How long does the communicator run on one charge?")
    assert hits, "no results"
    return f"{len(index.docs)} docs, {len(index.chunks)} chunks; top hit {hits[0].source}"


def mcp_tools():
    from relay_ops.mcp_client import ensure_server, get_mcp_tools
    state = ensure_server()
    names = sorted(asyncio.run(get_mcp_tools()))
    assert "search_knowledge_base" in names, f"tools found: {names}"
    return f"server {state}; tools: {', '.join(names)}"


def end_to_end():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from relay_ops.graph import build_system, new_turn

    async def run():
        rows = []
        for knowledge, via, q in [("list", "local", "How tall is Mount Everlore?"),
                                  ("rag", "local", "How long does a full recharge of the communicator take?"),
                                  ("rag", "local", "What is 847 times 23, plus 156?"),
                                  ("rag", "mcp", "What is the response time for a P1 incident?")]:
            graph = await build_system(knowledge, via, InMemorySaver(), InMemoryStore())
            out = await graph.ainvoke(new_turn(q), {"configurable": {"thread_id": f"check-{knowledge}-{via}", "user_id": "check"}})
            rows.append(f"\n         [{knowledge}/{via}] {q}\n           route={out['route']} -> {out['final']!r}")
        return "".join(rows)
    return asyncio.run(run())


if __name__ == "__main__":
    print("Relay Ops Assistant — setup check\n")
    if not check("Python 3.10 or newer", python_version, "create the venv with a newer Python (see README, step 2)"):
        sys.exit(1)
    if not check("Python packages installed", packages, "pip install -r requirements.txt"):
        sys.exit(1)
    check("Ollama running and model pulled", ollama, "install Ollama from https://ollama.com, open it, then: ollama pull qwen3:1.7b")
    if not problems:
        check("LLM answers", llm_replies, "check the Ollama app is running")
    check("RAG index builds (downloads 2 small models the first time)", rag_index, "check your internet connection for the first run")
    check("MCP server starts and lists its tools", mcp_tools, "see mcp_server.log in this folder")
    if "--full" in sys.argv and not problems:
        check("Agents run end to end", end_to_end, "re-run; if a small model misroutes, try RELAY_MODEL=qwen3:4b")
    print("\nAll good — start the app with:  streamlit run app.py" if not problems
          else f"\n{len(problems)} problem(s) to fix: {', '.join(problems)}")
    sys.exit(1 if problems else 0)
