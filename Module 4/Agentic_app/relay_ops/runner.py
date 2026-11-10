"""Run async LangGraph code from a synchronous UI such as Streamlit.

The graph is async (MCP tools only exist as coroutines), but Streamlit re-runs a plain script. A single
background event loop keeps the async clients (Ollama, MCP) on one loop for the lifetime of the app.
"""
import asyncio
import queue
import threading


class AsyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="graph-loop").start()

    def run(self, coro):
        """Run one coroutine to completion and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def iterate(self, agen):
        """Consume an async generator (e.g. graph.astream) as a normal generator, item by item, live."""
        q = queue.Queue()

        async def pump():
            try:
                async for item in agen:
                    q.put(("item", item))
            except BaseException as e:          # re-raised in the caller's thread
                q.put(("error", e))
            finally:
                q.put(("done", None))

        asyncio.run_coroutine_threadsafe(pump(), self.loop)
        while True:
            kind, value = q.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                return
