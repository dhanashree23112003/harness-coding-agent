# harness-coding-agent

> A production-shaped autonomous coding agent. Give it a task in plain English; it selects tools from a 54-tool registry, executes a multi-step plan, spawns isolated subagents when needed, and commits the result. Runs on a fully free stack.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-296_passing-brightgreen)
![Tools](https://img.shields.io/badge/tools-54_across_6_namespaces-orange)
![Stack](https://img.shields.io/badge/stack-LangGraph_%7C_MCP_%7C_pgvector-lightgrey)
![License](https://img.shields.io/badge/cost-%240_free_stack-success)

Built for the X-Arc hiring hackathon. The work is the harness around the model, not the model.

---

## What it demonstrates

| # | Property | How this build satisfies it |
|---|----------|------------------------------|
| **1** | 50+ tools, 4+ namespaces, model-driven selection | 54 tools across 6 MCP namespaces; semantic retrieval surfaces the relevant subset per step (recall@12 = **0.948** at 54 tools) |
| **2** | Real isolated subagent | Test-triage subagent with fresh context, code-enforced scoped toolset, own budget, and a typed structured return |
| **3** | Long-horizon 20+ call session | A single session completes 20+ tool calls; deterministic context compaction preserves the plan across the boundary |
| **4** | Production scaffolding | Backoff retries, token-bucket rate limiting, typed error hierarchy, JSON logging with correlation IDs, eval harness, 296 tests |
| **5** | Composable tool I/O | One tool's structured output feeds another's input (e.g. `ast.find_references` → `fs.read_file_range`) |

The headline piece is the **tool-retrieval layer**: rather than binding 54 schemas to the model on every step, each tool schema is embedded locally and stored in pgvector, and only the top-k relevant tools are passed to the model per step. See [SPEC.md](./SPEC.md) §5 and [MEMO.md](./MEMO.md).

---

## Stack

All free. No paid APIs.

| Layer | Choice |
|-------|--------|
| Orchestration | LangGraph (explicit state graph) |
| Tool transport | MCP servers, one per namespace, over stdio |
| Tool retrieval | pgvector + `all-MiniLM-L6-v2` embeddings (local, CPU) |
| Model | Groq free tier (`llama-3.1-8b-instant` default, configurable) |
| Language | Python 3.11+, pydantic-typed tool I/O |

---

## Quick start

**Prerequisites:** Python 3.11+, PostgreSQL with the pgvector extension, a Groq API key (free tier works).

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
#    then edit .env: set GROQ_API_KEY and DATABASE_URL

# 3. Smoke task (reads SPEC.md, returns one sentence)
python -m agent.main

# 4. The 20+ call long-horizon demo (fires context compaction)
python -m agent.main long_horizon
```

---

## Tests and evals

```bash
pytest tests/unit/ -v                         # unit tests, no API key needed
pytest tests/integration/ -v -m integration   # integration, needs Groq key + Postgres
python -m evals.retrieval.eval_recall          # retrieval recall@k (no API key needed)
```

The retrieval eval runs entirely on local embeddings and pgvector, so it produces the headline metric without any API calls.

---

## Make targets

```
make install            # pip install -e ".[dev]"
make test               # unit tests
make test-integration   # integration tests
make eval               # retrieval recall@k
make run                # default smoke task
make run-long           # 20+ call long-horizon demo
make docker-build       # docker build -t agent .
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | _required_ | Groq API key |
| `DATABASE_URL` | _required_ | PostgreSQL connection string with pgvector |
| `AGENT_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `CONTEXT_COMPACT_THRESHOLD` | `800` | Token estimate above which compaction fires |
| `RETRY_MAX_ATTEMPTS` | `3` | Max retry attempts on transient errors |
| `RETRY_BASE_DELAY` | `1.0` | Base delay (seconds) for exponential backoff |
| `RATE_LIMIT_RPM` | `30` | Outbound calls per minute (token-bucket rate) |
| `RATE_LIMIT_BURST` | `5` | Token-bucket burst size |

---

## Docker

```bash
docker build -t agent .
docker run --env-file .env agent              # smoke task
docker run --env-file .env agent long_horizon # long-horizon demo
```

---

## Optional: LangSmith tracing

Tracing is wired but optional (no-op when unconfigured). To enable, add to `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=harness-coding-agent
```

LangChain auto-enables tracing when these are set; every graph node, tool call, and subagent span appears in the LangSmith UI with no code changes.

---

## Architecture

```
src/agent/
  graph/             LangGraph state graph: plan, retrieve, act, tools, manage_context
  mcp_client/        MCP server discovery; per-invocation stdio sessions (AsyncExitStack)
  servers/           6 MCP servers: fs, git, ast, test, deps, ci
  retrieval/         pgvector store + sentence-transformer embeddings + top-k retriever + miss guard
  subagent/          Isolated subagent loop: scoped tools, own budget, typed SubagentResult
  resilience/        Exponential-backoff retry + token-bucket rate limiter
  errors.py          Typed error hierarchy (AgentError base)
  logging_config.py  JSON structured logging with per-session correlation IDs
evals/
  retrieval/         Labeled (goal, tool) pairs + recall@k
tests/
  unit/              Tool models, retrieval, compaction, backoff, subagent isolation
  integration/       Full agent run and subagent spawn-and-return paths
```

> Note on transport: MCP sessions are opened per invocation (a fresh connection per tool call) rather than held persistently. This was a deliberate stability choice on Windows after debugging an stdin-pipe inheritance issue; persistent sessions are noted as future work. See [MEMO.md](./MEMO.md).

The full architecture specification, including every design decision and the property-to-implementation contract, is in [SPEC.md](./SPEC.md).
