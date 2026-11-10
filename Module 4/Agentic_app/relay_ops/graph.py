"""The multi-agent system and its harness — Lab 1, Sections B to G, as one module.

   user ──► Supervisor ──► Research  (lookup_fact  OR  search_knowledge_base)  ──┐
               ▲     └───► Math      (run_python   OR  MCP calculate)           ──┤
               │                                                                 ▼
               └──────────────────────────── Writer ◄─────────────────────────────┘
   Supervisor (2nd visit) ──► END,  or ──► Approval (human-in-the-loop) ──► send_report ──► END

Two switches decide which tools the specialists get:
  knowledge : "list" -> lookup_fact (exact name match)   |  "rag" -> search_knowledge_base (hybrid + re-rank)
  tools_via : "local" -> in-process agent tools          |  "mcp" -> the same tools discovered from the MCP server
"""
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from . import tools
from .config import RISKY_TOOLS
from .llm import ainvoke_requiring_tool, get_llm
from .mcp_client import get_mcp_tools


# ================================================================== agents (Section B)
def build_react_agent(agent_tools, instructions: str, name: str, require_tool: bool = True):
    """A goal-based ReAct agent as a LangGraph graph: model -> (tools -> model)* -> END."""
    model = get_llm().bind_tools(agent_tools)

    async def call_model(state: MessagesState):
        msgs = [SystemMessage(instructions)] + state["messages"]
        used_a_tool = any(isinstance(m, ToolMessage) for m in state["messages"])
        if require_tool and not used_a_tool:
            reply = await ainvoke_requiring_tool(model, msgs)      # guardrail: no answering from memory
        else:
            reply = await model.ainvoke(msgs)
        return {"messages": [reply]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode(agent_tools, handle_tool_errors=True))   # tool errors become observations
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    return g.compile(name=name)


def tool_text(m: ToolMessage) -> str:
    """MCP tools return content blocks; local tools return a string. Normalise both to text."""
    if isinstance(m.content, str):
        return m.content
    return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in m.content)


def report(messages) -> str:
    """What a specialist hands back: its final answer plus the raw tool results as evidence."""
    final = (messages[-1].content or "").strip()
    evidence = [tool_text(m) for m in messages if isinstance(m, ToolMessage)]
    return (final + "\n" if final else "") + "Evidence from tools: " + " | ".join(evidence)


# ================================================================== shared state (Section C)
class OpsState(MessagesState):
    route: str            # which specialist the supervisor chose
    wants_email: bool     # did the user ask for the answer to be emailed?
    findings: str         # the specialist's report
    sources: list         # RAG chunks the research agent retrieved (file#chunk)
    final: str            # the writer's answer
    email_status: str     # "", "sent", "rejected"


def new_turn(text: str) -> dict:
    """Input for one user turn: the new message, with the per-turn fields reset."""
    return {"messages": [HumanMessage(text)], "final": "", "findings": "", "sources": [],
            "email_status": "", "wants_email": False, "route": ""}


MATH_WORDS = re.compile(r"\b(times|plus|minus|divided|multiply|multiplied|convert|percent|sum|average|total)\b|[%*/+=]", re.I)


def plain_text(answer: str) -> str:
    """Small models sometimes write LaTeX ($847 \\times 23$) even when told not to; keep the answer readable."""
    answer = re.sub(r"\$\s*([^$]*?)\s*\$", r"\1", answer)
    return answer.replace("\\times", "×").replace("\\div", "÷").replace("\\cdot", "·")


class Route(BaseModel):
    next: Literal["research", "math"] = Field(description="research: a question about facts, answered from documents. math: arithmetic on numbers written in the message.")


def last_user_message(state) -> str:
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def conversation(state, last_n: int = 6) -> str:
    """Recent turns as plain text, so agents can resolve words like 'that' (short-term memory)."""
    return "\n".join(f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {m.content}"
                     for m in state["messages"][-last_n:] if isinstance(m, (HumanMessage, AIMessage)) and m.content)


def user_prefs(cfg: RunnableConfig, store: BaseStore) -> dict:
    """Long-term memory (Section D): per-user preferences, shared by every conversation."""
    item = store.get(("users", cfg["configurable"].get("user_id", "anonymous")), "preferences")
    return item.value if item else {}


# ================================================================== the system
TOOL_PLANS = {
    # (knowledge, tools_via) -> (research tool name, math tool name)
    ("list", "local"): ("lookup_fact", "run_python"),
    ("rag", "local"): ("search_knowledge_base", "run_python"),
    ("list", "mcp"): ("lookup_fact", "calculate"),
    ("rag", "mcp"): ("search_knowledge_base", "calculate"),
}

MATH_HOW = {"run_python": "by calling run_python with Python code such as: answer = 340 / 28",
            "calculate": "by calling calculate with one arithmetic expression such as: 340 / 28"}


async def build_system(knowledge: str = "rag", tools_via: str = "local", checkpointer=None, store=None):
    """Assemble the multi-agent graph for one combination of knowledge source and tool transport."""
    research_name, math_name = TOOL_PLANS[(knowledge, tools_via)]
    available = await get_mcp_tools() if tools_via == "mcp" else \
        {t.name: t for t in (tools.lookup_fact, tools.search_knowledge_base, tools.run_python)}
    research_tool, math_tool = available[research_name], available[math_name]

    llm = get_llm()
    router = llm.with_structured_output(Route)

    research_agent = build_react_agent([research_tool],
        f"You are the RESEARCH agent. Find the facts needed for the task with {research_name}. "
        "Report the facts exactly as found, in one or two sentences. Do NOT do arithmetic.",
        name="research_agent")
    math_agent = build_react_agent([math_tool],
        f"You are the MATH agent. Solve the task {MATH_HOW[math_name]}. "
        "Never compute in your head. After you get the result, report it in one sentence.",
        name="math_agent")

    # ---------- nodes
    async def supervisor(state: OpsState) -> Command:
        if state.get("final"):                                   # 2nd visit: the writer has reported back
            if state.get("wants_email") and not state.get("email_status"):
                return Command(goto="approval")
            return Command(goto=END)
        decision = await router.ainvoke(                         # 1st visit: route the work
            "You route requests to specialist agents.\n"
            "- research: answers questions about facts (what, who, how many, how long, when) by looking them up in documents\n"
            "- math: computes an arithmetic expression whose numbers are written in the message (e.g. '847 times 23', '15% of 2,400')\n\n"
            f"Earlier conversation (context only):\n{conversation(state)}\n\n"
            f"LAST user message: {last_user_message(state)}\n\n"
            "Choose the specialist for the LAST user message only. If it does not contain numbers to compute with, choose research.")
        # Hybrid routing: rules back up the LLM where a small model is unreliable.
        #  1. math needs numbers to compute with. With none in the message, a math route is a mistake (in testing,
        #     "How many crew ...?" after an arithmetic turn went to math, which invented a number); with numbers AND
        #     an arithmetic word, a research route is (the writer then did the arithmetic itself, wrongly).
        #  2. the decision that leads to a HIGH-RISK action is a rule on the LAST message only. (Asked to decide it,
        #     the model re-read an email request from an earlier turn and asked for approval again.)
        text = last_user_message(state)
        has_numbers = bool(re.search(r"\d", text))
        route = "math" if has_numbers and MATH_WORDS.search(text) else decision.next
        if route == "math" and not has_numbers:
            route = "research"
        wants_email = bool(re.search(r"\b(email|e-mail|send|mail|forward)\b", text, re.I))
        return Command(goto=route, update={"route": route, "wants_email": wants_email})

    async def research(state: OpsState):
        task = f"Conversation so far:\n{conversation(state)}\n\nTask: find the facts needed for: {last_user_message(state)}"
        out = await research_agent.ainvoke({"messages": [HumanMessage(task)]})
        findings = report(out["messages"])
        sources = sorted(set(re.findall(r"\[source: ([^\]]+)\]", findings)))
        return {"findings": findings, "sources": sources}

    async def math(state: OpsState):
        task = f"Conversation so far:\n{conversation(state)}\n\nTask: {last_user_message(state)}"
        out = await math_agent.ainvoke({"messages": [HumanMessage(task)]})
        return {"findings": report(out["messages"]), "sources": []}

    async def writer(state: OpsState, config: RunnableConfig, *, store: BaseStore):
        prefs = user_prefs(config, store)
        reply = await llm.ainvoke([
            SystemMessage("You are the WRITER. Write the final answer to the user in at most two sentences, "
                          "using ONLY the findings. Copy numbers exactly from the evidence and never calculate anything yourself. "
                          "If the findings do not contain the answer, say you don't have that information. "
                          "Write plain text without LaTeX or formulas. Do not mention agents, tools, sources or the evidence."),
            HumanMessage(f"User request: {last_user_message(state)}\nFindings: {state['findings']}")])
        answer = plain_text(reply.content.strip())
        if state.get("sources"):
            answer += "\n\nRetrieved from: " + ", ".join(state["sources"])
        if prefs.get("sign_off"):
            answer += f"\n\n{prefs['sign_off']}"
        return {"final": answer, "messages": [AIMessage(answer)]}

    async def approval(state: OpsState, config: RunnableConfig, *, store: BaseStore):
        """Human-in-the-loop (Section E): pause before the high-risk tool and wait for a decision."""
        prefs = user_prefs(config, store)
        draft = {"tool": "send_report", "to": prefs.get("report_to", "ops@relay.example"),
                 "subject": "Relay Ops report", "body": state["final"]}
        assert draft["tool"] in RISKY_TOOLS
        decision = interrupt(draft)                              # <-- the graph pauses here
        if decision == "approve":
            await tools.send_report.ainvoke({k: draft[k] for k in ("to", "subject", "body")})
            return {"email_status": "sent"}
        return {"email_status": "rejected"}

    g = StateGraph(OpsState)
    g.add_node("supervisor", supervisor, destinations=("research", "math", "approval", END))
    g.add_node("research", research)
    g.add_node("math", math)
    g.add_node("writer", writer)
    g.add_node("approval", approval)
    g.add_edge(START, "supervisor")
    g.add_edge("research", "writer")
    g.add_edge("math", "writer")
    g.add_edge("writer", "supervisor")
    g.add_edge("approval", END)
    graph = g.compile(checkpointer=checkpointer, store=store)
    graph.agents = {"research_agent": research_agent, "math_agent": math_agent}   # kept for the UI's diagrams
    graph.tool_plan = {"research": research_name, "math": math_name, "via": tools_via}
    return graph

