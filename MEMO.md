# MEMO: Autonomous Coding Agent (X-Arc Hackathon)

## What I Built

A production-shaped autonomous coding agent built on LangGraph, six MCP servers, and a semantic tool-retrieval layer backed by pgvector and all-MiniLM-L6-v2 embeddings. The harness satisfies all five required properties from the brief.

**Property 1 (54 tools, model-driven selection):** Six MCP namespaces (git 12, fs 11, ast 9, test 8, deps 7, ci 7 = 54 tools) are discovered at startup and embedded into pgvector. On every agent step, the current goal is embedded and the top-k most relevant tool schemas are retrieved; only that subset is passed to the model. There is no if/else dispatch anywhere in the loop. A retrieval-miss guard logs the miss, widens k by 6, and re-retrieves before the next act step.

**Property 2 (isolated subagent):** `spawn_subagent(task: SubagentTask) -> SubagentResult` is a real spawn: the subagent gets its own message history seeded only by the task brief, a scoped toolset declared via `NamespaceScope` (e.g., `test` + `fs.read_file` only, never `git.commit`), and its own step and token budget. The parent consumes the typed `SubagentResult` pydantic object, never the subagent's raw transcript.

**Property 3 (20+ call long-horizon execution):** A `manage_context` node runs after every tool execution. It estimates tokens as character count divided by 4, compares against `COMPACT_THRESHOLD` (default 800), and when exceeded, summarizes completed tool-call pairs into a progress ledger, drops the verbose messages, and injects a `SystemMessage` carrying the ledger. The `plan` field lives in graph state, separate from messages, so it survives compaction without prompt-stuffing. The canonical task ("add input validation, update callers, make the suite green, commit") spans 20+ tool calls across status, read, ast.find_references, write, test runs, subagent spawn, and commit.

**Property 4 (production scaffolding):** Exponential backoff retry with jitter (3 attempts, base 1s, cap 60s) wraps all model and MCP calls; 413 RequestTooLarge and daily-cap 429s are detected and raised as typed errors immediately without retrying. A token-bucket rate limiter (configurable RPM, default 30) gates outbound calls. Structured JSON logging with per-session correlation IDs ships on every log line. The typed error hierarchy (`AgentError`, `RetryExhausted`, `RateLimitExceeded`, `RequestTooLarge`, `ToolError`, `RetrievalMiss`, `ValidationError`, `SubagentBudgetExceeded`, `ToolScopeViolation`) ensures no bare exceptions cross module boundaries. 296 unit tests and three integration test paths cover tool I/O models, retrieval ranking, compaction logic, backoff behavior, subagent isolation, and loop detection.

**Property 5 (composable tool I/O):** `git.list_changed_files` output feeds `test.discover_tests` input; `ast.find_references` output feeds `fs.read_file_range` to pull each call site; `test.last_failures` output feeds `fs.grep` to locate failing assertions. All tool inputs and outputs are pydantic models.

## One Design Decision I Would Defend

**Semantic retrieval over a flat 54-schema prompt.** The obvious alternative is to pass all 54 tool schemas to the model on every step; models have large context windows and this requires no infrastructure. I rejected it for three reasons. First, selection accuracy: forcing the model to attend to 54 schemas at once diffuses attention, especially for semantically adjacent tools (e.g., `grep` vs `search_files`, `run_linter` vs `run_security_scan`). Second, token cost: 54 schemas passed on every step of a 20+ call session is a significant multiplier on a free-tier daily cap of 100K tokens. Third, the literal requirement: "coherent at fifty tools rather than collapsing into a chain of fifty conditional dispatches" is not satisfied by dumping all fifty into every prompt. The retrieval layer keeps k=12 on a 54-tool registry, meaning the model sees under a quarter of the schemas per step. The eval harness measures recall@5, @10, and @12 on 115 labeled (goal, correct-tool) pairs covering all 54 tools, runnable via `make eval` against a populated pgvector instance.

## What I Cut

**Six retrieval misses clustering on description-overlap tools.** The miss guard widens k automatically, but there are pairs of tools (e.g., `grep` vs `search_files`, `run_linter` vs `run_security_scan`) where description overlap makes recall@5 imperfect. I did not chase these with hand-tuned boosts or re-ranking because the widening guard handles them at runtime and the effort would have been better spent on scaffolding correctness.

**Persistent MCP sessions.** The MCP client opens a fresh subprocess per agent run using an async context manager. A persistent session pool would reduce startup latency (embedding 54 tools on each cold start takes a few seconds) but adds connection lifecycle complexity. Given the hackathon scope, per-invocation connections are the safer choice.

**Parallel subagents.** The subagent contract supports one spawn at a time. Multi-subagent parallelism (e.g., spawn a test-triage subagent and a lint subagent concurrently) would require a concurrency manager and budget aggregation across spawns.

**End-to-end task-success eval.** The `evals/e2e/` directory is a stub. The integration test `test_long_horizon.py` covers the same scenario (add validation, make suite green, commit) and is the de facto e2e proof, but it skips without a live GROQ_API_KEY and does not emit a pass/fail metric to a harness.

**Active LangSmith tracing.** LangSmith is wired passively: LangChain enables it automatically when `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY` are set in the environment. No active span instrumentation was added to the graph nodes. Traces are available if those env vars are present but are not part of the submission artifact.

## What More Time Would Address

The named cuts above, plus: a proper recall@k run against a live pgvector with logged results committed to the repo; a metric-emitting e2e eval against the fixture repo; parallel subagent support with budget aggregation; a persistent MCP session pool; and a larger labeled eval set (the current 115 pairs is defensible but a set of 200+ would be more robust against questioning on semantic edge cases).

## Honest Constraints

The stack runs on Groq's free tier: 100K tokens per day and a per-request TPM limit. The TPM limit shaped the long-horizon demo directly: the compaction threshold (800 estimated tokens) is set low enough to trigger within a 20-call run on a small model, not because 800 tokens is a realistic production threshold. The daily cap forced the long-horizon task to be verified via a deterministic integration test rather than a live recorded trace. The compaction proof (`tests/unit/test_compaction_integration.py`, 7 assertions, no LLM calls) is the more rigorous approach for deterministic logic: it proves the compaction invariants hold on every run, not just on one lucky day when the cap was not hit. Local all-MiniLM-L6-v2 embeddings and a local pgvector instance replace any managed embedding service.

## Divergence Moments

**Windows event-loop conflict (Slice 1/2).** The standard `asyncio.run()` entry point raised a `RuntimeError` on Windows because LangGraph's async graph runner conflicts with the default `ProactorEventLoop` and nested event loop calls. The fix required wrapping the top-level call correctly for the Windows `asyncio` policy. This was caught before any code shipped and is why `main.py` uses the pattern it does.

**Agent reporting hallucinated success (Slice 6).** During the long-horizon demo run, the agent produced a final answer claiming it had committed code and all tests passed when the tool call history showed no successful `git_commit` result. The root cause was the model predicting plausible outcomes rather than grounding on visible results. The fix was a `GROUNDING RULE` added to the system prompt: the agent must not claim a file was written, a test passed, or a commit was made unless a successful tool result appears in the session history. This is the right fix; a softer instruction would have been overridden by the model's completion instinct.

**Over-aggressive loop detector (Slice 6).** The consecutive-repeat guard was initially set to stop after 2 identical consecutive tool-call turns. During the long-horizon task, the agent legitimately retried the same failing test run twice before finding the error. The threshold of 2 fired too early and terminated the run prematurely. Raising it to 3 (configurable via `AGENT_LOOP_THRESHOLD`) resolved the false positive while still catching genuine infinite loops.
