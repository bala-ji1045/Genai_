"""The model, plus the harness-level `tool_choice='required'` guardrail (Lab 1, A5)."""
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from . import config


@lru_cache(maxsize=4)
def get_llm(model: str = config.MODEL) -> ChatOllama:
    # temperature=0 -> repeatable; reasoning=False -> no long Qwen3 "thinking"; num_predict caps reply length
    return ChatOllama(model=model, base_url=config.OLLAMA_URL, temperature=0, reasoning=False, num_predict=512)


NUDGE = HumanMessage("You must call one of your tools now. Do not answer from memory.")


async def ainvoke_requiring_tool(model, messages, retries: int = 2):
    """Ollama ignores tool_choice, so the harness enforces it: nudge and retry until the model calls a tool."""
    reply = await model.ainvoke(messages)
    for _ in range(retries):
        if reply.tool_calls:
            break
        reply = await model.ainvoke(messages + [reply, NUDGE])
    return reply
