"""Relay Ops Assistant — Streamlit UI for the Module 4 multi-agent system.

Run:  streamlit run app.py
"""
import json
import urllib.request
import uuid

import pandas as pd
import streamlit as st
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from relay_ops import config, knowledge, rag, tools
from relay_ops.graph import build_system, new_turn, tool_text
from relay_ops.mcp_client import ensure_server, server_is_up
from relay_ops.runner import AsyncRunner

st.set_page_config(page_title="Relay Ops Assistant", page_icon="🛰️", layout="wide")

EXAMPLES = [
    "What is the battery capacity of the Zylarian Communicator?",
    "How long does a full recharge of the communicator take?",
    "What is 847 times 23, plus 156?",
    "How tall is Mount Everlore?",
    "Convert that height to feet. 1 meter is 3.28084 feet.",
    "What is the response time for a P1 incident?",
    "How many crew live at the Kelvorian Outpost? Email the summary to the ops lead.",
    "Who is the CEO of Relay Ops?",
]
KNOWLEDGE_LABELS = {"rag": "RAG — document library (hybrid search + re-rank)", "list": "List — Python dict (exact name match)"}
VIA_LABELS = {"local": "Local agent tools (in-process @tool)", "mcp": "MCP server (tools discovered over HTTP)"}


# ================================================================== long-lived objects (one per app process)
@st.cache_resource
def runtime():
    return {"runner": AsyncRunner(), "checkpointer": InMemorySaver(), "store": InMemoryStore(), "graphs": {}}


RT = runtime()


def get_graph(knowledge_mode: str, via: str):
    key = (knowledge_mode, via)
    if key not in RT["graphs"]:
        RT["graphs"][key] = RT["runner"].run(build_system(knowledge_mode, via, RT["checkpointer"], RT["store"]))
    return RT["graphs"][key]


def ollama_status():
    try:
        tags = json.load(urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=3))
        pulled = [m["name"] for m in tags["models"]]
        if any(n.startswith(config.MODEL) for n in pulled):
            return True, f"Ollama running · `{config.MODEL}` ready"
        return False, f"Ollama is running but `{config.MODEL}` is not pulled. Run: `ollama pull {config.MODEL}`"
    except OSError:
        return False, "Ollama is not running. Open the Ollama app (or run `ollama serve`)."


# ================================================================== session state
ss = st.session_state
ss.setdefault("thread_id", f"chat-{uuid.uuid4().hex[:8]}")
ss.setdefault("chat", [])          # [{"role", "content", "trace", "meta"}]
ss.setdefault("pending", None)     # an interrupt payload waiting for approve / reject
ss.setdefault("queued", None)      # an example question clicked in the sidebar

# ================================================================== sidebar: settings + harness controls
with st.sidebar:
    st.title("🛰️ Relay Ops")
    ok, msg = ollama_status()
    (st.success if ok else st.error)(msg)

    st.subheader("Agents' tools")
    knowledge_mode = st.radio("Knowledge source (research agent)", list(KNOWLEDGE_LABELS),
                              format_func=KNOWLEDGE_LABELS.get, key="knowledge_mode")
    via = st.radio("Tools come from", list(VIA_LABELS), format_func=VIA_LABELS.get, key="via")
    if via == "mcp":
        if server_is_up():
            st.caption(f"✅ MCP server up at `{config.MCP_URL}`")
        else:
            st.caption("MCP server not running yet — it will start on the first question (first start loads the RAG models, ~30 s).")

    st.subheader("Long-term memory (store)")
    user_id = st.text_input("User id", value="asha")
    ns = ("users", user_id)
    if RT["store"].get(ns, "preferences") is None:
        RT["store"].put(ns, "preferences", {"report_to": "ops-lead@relay.example", "sign_off": "— Relay Ops Assistant"})
    prefs = RT["store"].get(ns, "preferences").value
    report_to = st.text_input("Reports go to", value=prefs.get("report_to", ""))
    sign_off = st.text_input("Sign-off", value=prefs.get("sign_off", ""))
    if st.button("Save preferences"):
        RT["store"].put(ns, "preferences", {"report_to": report_to, "sign_off": sign_off})
        st.toast("Saved to the long-term store — every conversation of this user will use it.")

    st.subheader("Short-term memory (checkpointer)")
    st.caption(f"Conversation thread: `{ss.thread_id}`")
    if st.button("🆕 New conversation"):
        ss.thread_id, ss.chat, ss.pending = f"chat-{uuid.uuid4().hex[:8]}", [], None
        st.rerun()

    st.subheader("Limits")
    recursion_limit = st.slider("Max graph steps (recursion_limit)", 2, 50, config.RECURSION_LIMIT)

    st.subheader("Try these")
    for q in EXAMPLES:
        if st.button(q, key=f"ex-{q}", width="stretch"):
            ss.queued = q

run_cfg = {"configurable": {"thread_id": ss.thread_id, "user_id": user_id}, "recursion_limit": recursion_limit}


# ================================================================== running the graph with a live trace
def short(v, n=160):
    s = " ".join(str(v).split())
    return s if len(s) <= n else s[:n] + "…"


def describe(namespace, chunk) -> list:
    """Turn one `astream(..., subgraphs=True, stream_mode='updates')` event into readable trace lines."""
    lines = []
    inside = namespace[0].split(":")[0] if namespace else None
    for node, update in chunk.items():
        if node == "__interrupt__":
            lines.append("⏸️ **approval** — graph paused, waiting for a human decision")
            continue
        update = update or {}
        if inside:                                            # a step inside a specialist sub-agent
            for m in update.get("messages", []):
                if getattr(m, "tool_calls", None):
                    for c in m.tool_calls:
                        lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ `{inside}` calls **{c['name']}** `{json.dumps(c['args'])}`")
                elif isinstance(m, ToolMessage):
                    lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ `{inside}` ← tool returned: {short(tool_text(m), 300)}")
                elif m.content:
                    lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ `{inside}` says: {short(m.content)}")
        else:                                                 # a step of the main graph
            written = {k: v for k, v in update.items() if k != "messages"}
            if node == "supervisor" and not written:
                lines.append("**supervisor** — writer reported back → finish (or ask for approval)")
            elif node == "supervisor":
                lines.append(f"**supervisor** → routes to **{written.get('route')}** · wants_email={written.get('wants_email')}")
            elif node in ("research", "math"):
                lines.append(f"**{node}** → findings: {short(written.get('findings', ''), 220)}")
            elif node == "writer":
                lines.append("**writer** → final answer written")
            elif node == "approval":
                lines.append(f"**approval** → email_status = `{written.get('email_status')}`")
    return lines


def run_graph(graph, graph_input):
    """Stream one run into a status box; return (trace lines, interrupt payload or None)."""
    trace, pending = [], None
    with st.status("Agents at work…", expanded=True) as box:
        try:
            events = graph.astream(graph_input, run_cfg, stream_mode="updates", subgraphs=True)
            for namespace, chunk in RT["runner"].iterate(events):
                if "__interrupt__" in chunk and not namespace:
                    pending = chunk["__interrupt__"][0].value
                for line in describe(namespace, chunk):
                    trace.append(line)
                    box.markdown(line)
            box.update(label="Paused for approval" if pending else "Done", state="complete", expanded=False)
        except GraphRecursionError:
            trace.append(f"🛑 Stopped by the harness: more than {recursion_limit} steps (recursion_limit).")
            box.update(label="Stopped by step limit", state="error")
        except Exception as e:
            trace.append(f"❌ {type(e).__name__}: {e}")
            box.update(label="Error", state="error")
    return trace, pending


def finish_turn(graph, trace, pending):
    values = graph.get_state(run_cfg).values
    meta = f"route: {values.get('route', '?')} · knowledge: {knowledge_mode} · tools: {via}"
    if values.get("email_status"):
        meta += f" · email: {values['email_status']}"
    errored = trace and trace[-1].startswith(("🛑", "❌"))
    content = trace[-1] if errored else (values.get("final") or "_(no answer)_")
    ss.chat.append({"role": "assistant", "content": content, "trace": trace, "meta": meta})
    ss.pending = pending


def handle_question(question: str):
    ss.chat.append({"role": "user", "content": question})
    with st.spinner("Preparing agents (the MCP server may be starting)…"):
        try:
            graph = get_graph(knowledge_mode, via)
        except Exception as e:
            ss.chat.append({"role": "assistant", "content": f"❌ Could not build the agents: {e}", "trace": [], "meta": ""})
            return
    trace, pending = run_graph(graph, new_turn(question))
    finish_turn(graph, trace, pending)


def handle_decision(decision: str):
    graph = get_graph(knowledge_mode, via)
    trace, pending = run_graph(graph, Command(resume=decision))
    values = graph.get_state(run_cfg).values
    status = values.get("email_status", "?")
    text = (f"📧 Email **sent** to `{ss.pending['to']}`." if status == "sent" else "🚫 Email **rejected** — nothing was sent.")
    ss.chat.append({"role": "assistant", "content": text, "trace": trace, "meta": f"approval: {decision}"})
    ss.pending = pending


# ================================================================== main area
st.header("Relay Ops Assistant")
st.caption("Supervisor → Research / Math → Writer → Supervisor · LangGraph + Ollama · memory, human approval, MCP, RAG")

tab_chat, tab_kb, tab_lab, tab_graph, tab_hist, tab_out = st.tabs(
    ["💬 Chat", "📚 Knowledge base", "🔎 Retrieval lab", "🧭 Graph", "📜 History", f"📤 Outbox ({len(tools.OUTBOX)})"])

with tab_chat:
    for msg in ss.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander(f"Trace · {msg.get('meta', '')}"):
                    for line in msg["trace"]:
                        st.markdown(line)

    question = st.chat_input("Ask about the Relay Ops world, do arithmetic, or ask to email the answer…",
                             disabled=ss.pending is not None)
    question = question or ss.queued
    ss.queued = None
    if question and ss.pending is None:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            handle_question(question)
        st.rerun()

    if ss.pending:
        with st.container(border=True):
            st.warning("**Human-in-the-loop:** the agents want to run a HIGH-RISK tool. Approve?")
            st.json(ss.pending)
            c1, c2 = st.columns(2)
            if c1.button("✅ Approve and send", type="primary", width="stretch"):
                handle_decision("approve")
                st.rerun()
            if c2.button("🚫 Reject", width="stretch"):
                handle_decision("reject")
                st.rerun()

with tab_kb:
    left, right = st.columns(2)
    with left:
        st.subheader("List knowledge base")
        st.caption("`relay_ops/knowledge.py` — used by `lookup_fact`. Finds an entry only if the query contains its name.")
        st.dataframe(pd.DataFrame([{"entry": k, "fact": v} for k, v in knowledge.KNOWLEDGE_BASE.items()]),
                     hide_index=True, width="stretch")
    with right:
        st.subheader("RAG document library")
        st.caption(f"`{config.DOCS_DIR.relative_to(config.APP_DIR)}/` — used by `search_knowledge_base`. "
                   "Add a file and both the app and the MCP server re-index it automatically.")
        upload = st.file_uploader("Add a .md or .txt document", type=["md", "txt"])
        if upload is not None and st.button("Add to library"):
            (config.DOCS_DIR / upload.name).write_bytes(upload.getvalue())
            st.success(f"Added {upload.name}. It will be indexed on the next search.")
        for doc in rag.load_documents():
            with st.expander(doc.metadata["source"]):
                st.markdown(doc.page_content)

with tab_lab:
    st.subheader("Look inside the retriever")
    st.caption("The same pipeline as the Module 3 lab: chunk → dense (FAISS) + sparse (BM25) → RRF fusion → cross-encoder re-rank.")
    q = st.text_input("Query", value="How long does the communicator run on one charge?")
    if st.button("Search", type="primary") and q:
        with st.spinner("Building the index / loading models (first time only)…"):
            index = rag.get_index()
        st.caption(f"{len(index.docs)} documents → {len(index.chunks)} chunks "
                   f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
        hybrid = index.hybrid(q, k=6, top_k=6)
        stages = [("Dense — FAISS", index.dense(q)), ("Sparse — BM25", index.sparse(q)),
                  ("Hybrid — RRF", hybrid), ("Re-ranked — cross-encoder (what the agent sees)", index.rerank(q, hybrid))]
        cols = st.columns(2)
        for i, (title, hits) in enumerate(stages):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame([{"source": h.source, "score": round(h.score, 4), "text": short(h.text, 120)}
                                           for h in hits]), hide_index=True, width="stretch")

with tab_graph:
    st.subheader("The system as a LangGraph graph")
    st.caption("Dotted edges are decided at run time (the supervisor's Command(goto=...)); solid edges are fixed.")
    if st.button("Draw the graphs"):
        try:
            graph = get_graph(knowledge_mode, via)
        except Exception as e:
            st.error(f"Could not build the agents: {e}")
        else:
            st.write("Tools in use:", graph.tool_plan)
            for title, g in [("Main graph", graph)] + [(n, a) for n, a in graph.agents.items()]:
                st.markdown(f"**{title}**")
                try:
                    st.image(g.get_graph().draw_mermaid_png())        # uses mermaid.ink, needs internet
                except Exception:
                    st.code(g.get_graph().draw_mermaid(), language="text")

with tab_hist:
    st.subheader("Checkpoint history of this conversation")
    st.caption("Every step is checkpointed — the raw material for debugging, replay and evaluation. Newest first.")
    graph = next(iter(RT["graphs"].values()), None)        # all modes share one checkpointer, so any graph can read it
    if graph is None:
        st.info("Ask a question first.")
    else:
        rows = [{"step": s.metadata.get("step"), "next": ", ".join(s.next) or "END",
                 "route": s.values.get("route", ""), "messages": len(s.values.get("messages", [])),
                 "final": short(s.values.get("final", ""), 60), "email": s.values.get("email_status", "")}
                for s in graph.get_state_history({"configurable": {"thread_id": ss.thread_id}})]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tab_out:
    st.subheader("Outbox")
    st.caption("`send_report` writes here instead of a real mail server. It only runs after a human approves.")
    if tools.OUTBOX:
        st.dataframe(pd.DataFrame(tools.OUTBOX[::-1]), hide_index=True, width="stretch")
    else:
        st.info("No emails sent yet. Try: “How many crew live at the Kelvorian Outpost? Email the summary to the ops lead.”")
