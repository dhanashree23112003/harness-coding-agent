# harness-coding-agent

A production-shaped autonomous coding agent built for the X-Arc hiring hackathon. It demonstrates model-driven tool selection over 50+ tools across 6 MCP namespaces (Property 1), real subagent isolation with structured returns (Property 2), explicit context compaction over 20+ tool-call sessions (Property 3), and production scaffolding including retries, rate limiting, typed errors, and structured logging (Property 4).

## Prerequisites

- Python 3.11+
- PostgreSQL with the `pgvector` extension enabled
- A Groq API key (free tier works; `llama-3.1-8b-instant` is the default model)

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env: set GROQ_API_KEY and DATABASE_URL

# 3. Run the default smoke task (reads SPEC.md, returns one sentence)
python -m agent.main

# 4. Run the 20+ call long-horizon demo (fires context compaction)
python -m agent.main long_horizon
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key |
| `DATABASE_URL` | required | PostgreSQL connection string with pgvector |
| `AGENT_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `CONTEXT_COMPACT_THRESHOLD` | `800` | Token estimate above which compaction fires |
| `RETRY_MAX_ATTEMPTS` | `3` | Max retry attempts on transient errors |
| `RETRY_BASE_DELAY` | `1.0` | Base delay (seconds) for exponential backoff |
| `RATE_LIMIT_RPM` | `30` | Outbound calls per minute (token bucket rate) |
| `RATE_LIMIT_BURST` | `5` | Token bucket burst size |

## Tests

```bash
# Unit tests only (no API key required)
pytest tests/unit/ -v

# Integration tests (requires GROQ_API_KEY and running PostgreSQL)
pytest tests/integration/ -v -m integration

# Retrieval recall@k evaluation
python -m evals.retrieval.eval_recall
```

## Make targets

```bash
make install          # pip install -e ".[dev]"
make test             # unit tests
make test-integration # integration tests
make eval             # retrieval recall@k eval
make run              # default smoke task
make run-long         # 20+ call long-horizon demo
make docker-build     # docker build -t agent .
```

## Docker

```bash
docker build -t agent .
docker run --env-file .env agent
# For the long-horizon demo:
docker run --env-file .env agent long_horizon
```

## Optional: LangSmith tracing

Add to `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=harness-coding-agent
```

LangChain auto-enables tracing when these are set. Every graph node, tool call, and subagent span will appear in the LangSmith UI with no code changes.

## Architecture

```
src/agent/
  graph/           LangGraph state graph (plan, retrieve, act, tools, manage_context)
  mcp_client/      MCP server discovery and persistent stdio sessions
  servers/         6 MCP servers: fs, git, ast, test, deps, ci
  retrieval/       pgvector store + sentence-transformer embeddings + top-k retriever
  subagent/        Isolated subagent loop with scoped tools and typed result
  resilience/      Exponential-backoff retry + token-bucket rate limiter
  errors.py        Typed error hierarchy (AgentError base)
  logging_config.py JSON structured logging with per-session correlation IDs
```

See `SPEC.md` for the full architecture specification.
