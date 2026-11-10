# Relay Ops Assistant — a multi-agent application

This app takes the multi-agent system from **`Lab/1_Multi_Agent_System_End_to_End.ipynb`** and turns it into a
web application with a chat window. Everything runs on your own computer, and you don't need an API key.

It adds two things the notebook doesn't have:

1. **A choice of where tools come from.** The same tools can run as **local agent tools** (`@tool` functions
   inside the app) or come from an **MCP server** (a separate process that the app finds over HTTP). You switch
   between them with one click.
2. **A choice of knowledge source.** Besides the notebook's **list** knowledge base (a Python dict with exact name
   matching), there is a **RAG** knowledge base: a folder of documents searched with the Module 3 pipeline
   (chunking, FAISS dense search, BM25, Reciprocal Rank Fusion, cross-encoder re-ranking).

```
                       ┌──────────────┐ ◄──────────────────────────────────┐
    you (Streamlit) ─► │  Supervisor  │ ─► END   or ─► Approval ─► END      │
                       └──────┬───────┘     (a human approves the email)    │
                routes to ONE │ specialist                                  │
             ┌────────────────┴─────────────────┐                           │
             ▼                                  ▼                           │
   ┌───────────────────────┐        ┌───────────────────────┐               │
   │    Research agent     │        │      Math agent       │               │
   │ lookup_fact     (list)│        │ run_python    (local) │               │
   │ search_knowledge_base │        │ calculate       (MCP) │               │
   │                  (RAG)│        │                       │               │
   └───────────┬───────────┘        └───────────┬───────────┘               │
               └────────────────┬───────────────┘                           │
                                ▼                                           │
                          ┌──────────┐       final answer                   │
                          │  Writer  │ ─────────────────────────────────────┘
                          └──────────┘
  Harness: checkpointer (short-term memory) · store (long-term memory) · interrupt() (human-in-the-loop)
           · MCP tool discovery · recursion limit · live trace · checkpoint history
```

---

## 1. What you need

| | Requirement | How to check |
|---|---|---|
| 1 | **Python 3.10 or newer** (the `mcp` package does not support 3.9) | `python3 --version` |
| 2 | **Ollama**, a free app that runs the language model on your computer | open http://localhost:11434 in a browser; it should say *"Ollama is running"* |
| 3 | About **4 GB of free disk space** for the language model, PyTorch and two small RAG models | — |
| 4 | An internet connection **for the first run only** (to download packages and models) | — |

8 GB of RAM is enough. The default model is `qwen3:1.7b`.

---

## 2. Set up (one time, about 10 minutes)

All commands are run in a terminal, **from inside this folder** (`Module 4/Agentic_app`).

### Step 1 — Install Ollama and download the model

1. Download Ollama from **https://ollama.com**, install it and open it. On macOS and Windows it runs in the menu bar or tray.
2. Download the model:
   ```bash
   ollama pull qwen3:1.7b
   ```
   If you have 16 GB of RAM or more, `qwen3:4b` follows instructions noticeably better. See [Settings](#8-settings) to switch.

### Step 2 — Create a virtual environment

macOS / Linux:
```bash
cd "Module 4/Agentic_app"
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
cd "Module 4\Agentic_app"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

> If `python3 --version` shows 3.9 or older, use a newer interpreter to create the venv. For example, with Anaconda:
> `/opt/anaconda3/bin/python3 -m venv .venv`. With Homebrew: `brew install python@3.12`, then `python3.12 -m venv .venv`.

### Step 3 — Install the packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes the first time because it downloads PyTorch, which `sentence-transformers` needs.

### Step 4 — Check the setup

```bash
python check_setup.py
```

Every line should say `[ OK ]`. If one says `[FAIL]`, the next line tells you how to fix it. The first run downloads
the two RAG models (about 100 MB) and starts the MCP server once, so it takes about a minute.

To also run the agents end to end on four test questions, add `--full`:

```bash
python check_setup.py --full
```

---

## 3. Run the app

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. If it doesn't, open that address yourself.
To stop the app, press `Ctrl+C` in the terminal.

---

## 4. Try it yourself: a guided test

Work through these in order. Each question is also a button under **"Try these"** in the sidebar.
After each answer, open **"Trace"** under it to see which agent ran and which tool it called.

Leave the sidebar on **Knowledge source = RAG** and **Tools come from = Local agent tools** to start.

| # | Ask this | What you should see | What it shows |
|---|---|---|---|
| 1 | *What is the battery capacity of the Zylarian Communicator?* | **3,200 mAh**, followed by `Retrieved from: zylarian_communicator.md#…` | Supervisor routes to **research**; RAG finds the right chunk and the answer cites it |
| 2 | *How long does a full recharge of the communicator take?* | **2 hours** | A fact that exists **only** in the documents, not in the list knowledge base |
| 3 | *What is 847 times 23, plus 156?* | **19,637** | Supervisor routes to **math**; the trace shows `run_python` with code the model wrote (program-of-thought) |
| 4 | *How tall is Mount Everlore?* then *Convert that height to feet. 1 meter is 3.28084 feet.* | **8,850 m**, then **about 29,035 feet** | **Short-term memory**: the second question says "that", and the checkpointer remembers the first answer |
| 5 | *How many crew live at the Kelvorian Outpost? Email the summary to the ops lead.* | The answer (**46**), then an **approval box** showing the draft email | **Human-in-the-loop**: the graph pauses before the high-risk `send_report` tool. Click **Approve** and the email appears on the **Outbox** tab. Ask again and click **Reject**: nothing is sent |
| 6 | *Who is the CEO of Relay Ops?* | Says it doesn't have that information | RAG always returns *something*; the grounded writer prompt stops the system from inventing an answer |

Then flip the switches in the sidebar and ask the same questions again. Click **🆕 New conversation** first, so that
earlier answers in the conversation don't help the agents:

| Switch | Ask | What changes |
|---|---|---|
| **Knowledge source → List** | Q2 (*recharge*) | The answer is now *"I don't have that information"*: the list has only one sentence per entry. Q1 still works |
| **Knowledge source → List** | *What's the battery size on the comm device?* | Fails: `lookup_fact` needs the exact name. Switch back to **RAG** and it works, because dense search matches meaning, not exact words |
| **Tools come from → MCP server** | Q1 and Q3 | Same answers, but the trace now shows `search_knowledge_base` and `calculate` **coming from the MCP server**. The first question takes about 30 s longer while the server starts |
| **Long-term memory**: change *Sign-off*, click **Save preferences**, click **New conversation** | any question | The new sign-off appears, even in a brand-new conversation, because it lives in the **store**, not in the conversation |
| **Limits**: set *Max graph steps* to **2** | any question | The run stops with *"Stopped by the harness"*: the recursion limit protects you from an agent that loops forever |

### The other tabs

| Tab | What it shows |
|---|---|
| **📚 Knowledge base** | Both knowledge sources side by side. **Upload your own `.md` or `.txt` file** and ask about it straight away: the app and the MCP server both re-index automatically |
| **🔎 Retrieval lab** | Type a query and compare the four retrieval stages (dense, BM25, hybrid, re-ranked) with their scores. Try *"rate limit"* versus *"how long does it last on one charge"* |
| **🧭 Graph** | Diagrams of the main graph and of each specialist's own ReAct graph (they need internet; without it you get Mermaid text instead) |
| **📜 History** | Every checkpoint of the current conversation: the raw material for debugging and replay |
| **📤 Outbox** | Emails that `send_report` "sent". Each one was approved by a human |

> **About small models.** `qwen3:1.7b` is small. Now and then it will route a question to the wrong agent or word an
> answer oddly. The trace shows you exactly where that happened, which is part of the lesson. If it happens a lot, use `qwen3:4b`.

---

## 5. Using the MCP server on its own

The MCP server is an ordinary MCP server. Any MCP client can use it, not just this app.

```bash
# HTTP (what the app uses). The app starts this for you automatically if nothing is listening on port 8765.
python mcp_server/knowledge_server.py

# stdio (what Claude Desktop and most IDEs use)
python mcp_server/knowledge_server.py --transport stdio
```

| It exposes | Name | What it does |
|---|---|---|
| Tool | `lookup_fact` | Looks up the list knowledge base by name |
| Tool | `search_knowledge_base` | RAG search over `data/docs/`, returns passages with their source |
| Tool | `calculate` | Safe arithmetic, e.g. `340 / 28` |
| Resource | `kb://{topic}` | One list entry |
| Resource | `docs://list` | The names of the documents in the library |

**Inspect it in a browser** with the official MCP Inspector (needs Node.js):
```bash
npx @modelcontextprotocol/inspector .venv/bin/python mcp_server/knowledge_server.py --transport stdio
```

**Add it to Claude Desktop.** Put this in `claude_desktop_config.json`, using full paths:
```json
{
  "mcpServers": {
    "relay-ops": {
      "command": "/full/path/to/Agentic_app/.venv/bin/python",
      "args": ["/full/path/to/Agentic_app/mcp_server/knowledge_server.py", "--transport", "stdio"]
    }
  }
}
```

The same server, unchanged, now serves two different clients. That is the point of MCP.

---

## 6. How the code is organised

```
Agentic_app/
├── app.py                     Streamlit UI: chat, live trace, approval box, tabs
├── check_setup.py             Checks the setup and tells you how to fix problems
├── requirements.txt
├── data/docs/                 The RAG document library (add your own .md / .txt here)
├── mcp_server/
│   └── knowledge_server.py    MCP server: lookup_fact, search_knowledge_base, calculate
└── relay_ops/                 The system itself; no UI code in here
    ├── config.py              Model name, RAG settings, MCP port, risky-tool list
    ├── llm.py                 ChatOllama + the "must call a tool" guardrail
    ├── knowledge.py           Knowledge source 1: the list (Python dict)
    ├── rag.py                 Knowledge source 2: chunk → FAISS + BM25 → RRF → re-rank
    ├── tools.py               Local agent tools: lookup_fact, search_knowledge_base, run_python, send_report
    ├── mcp_client.py          Starts the MCP server and discovers its tools
    ├── graph.py               Agents, shared state, supervisor, writer, approval: the LangGraph system
    └── runner.py              Runs the async graph from Streamlit's synchronous script
```

Where each idea from the notebooks lives:

| Idea | Notebook | File / function |
|---|---|---|
| Tools and their schemas, program-of-thought | Lab 1, Section A | `relay_ops/tools.py` |
| "Must call a tool" guardrail (`tool_choice` that Ollama ignores) | Lab 1, A5 | `llm.py → ainvoke_requiring_tool` |
| ReAct agent as a LangGraph graph | Lab 1, B3 | `graph.py → build_react_agent` |
| Shared state, supervisor with structured output, writer | Lab 1, Section C | `graph.py → OpsState, supervisor, writer` |
| Short-term memory (checkpointer) and long-term memory (store) | Lab 1, Section D | `app.py → runtime()`, `graph.py → user_prefs` |
| Human-in-the-loop with `interrupt()` | Lab 1, Section E | `graph.py → approval`, `app.py → handle_decision` |
| MCP tool discovery | Lab 1 F, Lab 2 | `mcp_client.py`, `mcp_server/knowledge_server.py` |
| Streaming trace, recursion limit, checkpoint history | Lab 1, Section G | `app.py → run_graph`, History tab |
| Chunking, dense, sparse, hybrid, re-ranking | Module 3 RAG lab | `rag.py → RAGIndex` |

You can also use the system without the UI, for example in a notebook:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from relay_ops.graph import build_system, new_turn

graph = await build_system("rag", "local", InMemorySaver(), InMemoryStore())
out = await graph.ainvoke(new_turn("How long does a full recharge of the communicator take?"),
                          {"configurable": {"thread_id": "t1", "user_id": "asha"}})
print(out["final"])
```

---

## 7. Design notes: what testing taught us

These changes were made after watching the app fail. Each one is a small lesson in its own right.

| Problem seen in testing | Fix | Where |
|---|---|---|
| RAG missed *"battery capacity of the Zylarian Communicator"*: the chunk with the answer (`## Power — It has a 3,200 mAh battery`) never names the product | **Contextual chunk headers**: every chunk starts with its document's title | `rag.py → chunk_documents` |
| After an arithmetic turn, *"How many crew…?"* was routed to **math**, which invented a number | **Hybrid routing**: math needs numbers in the message, otherwise go to research. Digits plus an arithmetic word (*times*, *convert*, *%*…) always go to math | `graph.py → supervisor` |
| After *"…email the summary"*, the **next** question also asked for approval: the LLM re-read the old request in the history | The decision that leads to a high-risk action is a **rule on the latest message only** | `graph.py → supervisor` |
| The writer wrote LaTeX (`$847 \times 23$`) despite being told not to | Clean the text in code instead of trusting the prompt | `graph.py → plain_text` |

The general lesson: **with a small model, put rules around the decisions that matter**, and keep the LLM for the
parts that rules can't do. That is what a harness is for.

---

## 8. Settings

Set these as environment variables before `streamlit run app.py`, for example `RELAY_MODEL=qwen3:4b streamlit run app.py`.

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_MODEL` | `qwen3:1.7b` | The Ollama model. Pull it first with `ollama pull …` |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama runs |
| `RELAY_MCP_PORT` | `8765` | Port of the MCP server |
| `RELAY_DOCS_DIR` | `data/docs` | Folder of RAG documents |

Chunk size, `top_k` and the embedding and re-ranking models are in `relay_ops/config.py`.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar says *"Ollama is not running"* | Open the Ollama app, or run `ollama serve` in another terminal |
| *"… is not pulled"* | `ollama pull qwen3:1.7b` |
| `pip install` fails on `mcp` | Your Python is older than 3.10. Re-create the venv (step 2) |
| MCP mode: *"The MCP server exited during start-up"* | Read `mcp_server.log` in this folder. Often port 8765 is in use: set `RELAY_MCP_PORT=8766` |
| MCP mode still uses old code after you edit the server | A server started by an earlier run is still alive. Stop it: `pkill -f knowledge_server.py` (Windows: close the Python process in Task Manager) |
| The first question is slow | The first RAG search loads the embedding and re-ranking models (about 10–30 s). Later questions are fast |
| Answers are sometimes wrong or routed to the wrong agent | A 1.7B model is small. Read the Trace to see where it went wrong, or try `RELAY_MODEL=qwen3:4b` |
| *"Stopped by the harness"* | The *Max graph steps* slider is too low. 25 is the default |

Memory (conversations, preferences, outbox) is kept **in RAM**, so it resets when you restart the app. To keep it, swap
`InMemorySaver` for `SqliteSaver` from `langgraph-checkpoint-sqlite` in `app.py → runtime()`.

---

## 10. Exercises

1. **Research, then math.** *"How many days can the Orbital Relay Station keep operating?"* needs a fact **and** a
   calculation. Add a route that sends research findings to the math agent before the writer.
2. **A new MCP tool.** Add a `convert_units` tool to `mcp_server/knowledge_server.py`. Restart, switch to MCP mode,
   and check the trace: the app found the new tool without any change to the app's code.
3. **Your own documents.** Upload a document about your own project on the Knowledge base tab and ask about it.
   Then use the Retrieval lab to compare which retriever finds it best.
4. **Persistent memory.** Replace `InMemorySaver` with `SqliteSaver` so conversations survive a restart.
5. **A hosted model.** Replace `ChatOllama` in `relay_ops/llm.py` with `ChatOpenAI` or `ChatAnthropic`. How many lines change?
