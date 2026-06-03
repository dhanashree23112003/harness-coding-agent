# SPEC: Autonomous Coding Agent (X-Arc Hackathon)

> This document is the build plan and the architectural record. It is committed at the repo root and is the source from which MEMO.md is written. Every section below maps to one of the five required properties or to a deployment concern. Build in the order of Section 9. Do not build breadth-first.

---

## 1. What this is

An autonomous coding agent: an LLM in a control loop that operates on a real repository through typed tools, plans multi-step changes, spawns isolated subagents for bounded sub-tasks, and runs under production scaffolding (observability, retries, rate limiting, typed errors, tests, eval harness).

The domain is deliberate. X-Arc builds "the engineering layer between frontier models and the systems that depend on them." They grade the harness, not domain knowledge. A coding agent puts the harness in the foreground and leans on engineering discipline rather than retrieval breadth. It also mirrors X-Arc's own work, which is the point.

Non-goal: building a better code-generation model. The model is a dependency. The work is the harness around it.

## 2. Stack and why each piece earns its place

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Orchestration | LangGraph | Explicit state graph makes the control loop and context strategy legible in code (Property 3). A hand-rolled while-loop would hide the plan-coherence logic the brief asks to see. Also on Deriv's stack. |
| Tool transport | MCP servers, one per namespace | Tools live behind a protocol, not baked into the agent (Property 1). Closes the standing MCP gap and gives a real client-discovers-tools story. |
| Tool retrieval | pgvector + sentence-transformer embeddings over tool schemas | 50 schemas will not fit in every prompt. Semantic retrieval of the relevant subset is the honest answer to "coherent at fifty rather than a chain of fifty dispatches." Reuses the embedding + pgvector pattern already shipped in Smart AI Inbox. |
| Observability | LangSmith | Traces every node, tool call, and subagent span (Property 4). On Deriv's stack. |
| Language | Python 3.11+ | Typed (pydantic models for tool I/O), matches Deriv, fastest path to real tests. |

The stack is chosen so one build satisfies X-Arc and rehearses Deriv's exact stack (LangGraph, pgvector, LangSmith, MCP) at the same time.

## 3. Property-to-implementation map

This table is the contract. If a row is not satisfied, the build is not done.

| # | Brief requirement | Where it lives in this build | Pass condition |
|---|---|---|---|
| 1 | 50+ tools, 4+ namespaces, model-driven, coherent registry | 6 MCP servers (Section 4) + retrieval layer (Section 5) | Model selects tools after semantic retrieval; no hand-routed if/else dispatch anywhere in the loop |
| 2 | Real subagent: isolated context, scoped toolset, structured return | `subagent` tool (Section 6) | Subagent runs in its own context with its own tool subset and returns a typed object the parent consumes |
| 3 | Long-horizon 20+ calls, context strategy in code | Context manager node in the graph (Section 7) | A single session task completes 20+ tool calls; compaction logic is explicit code, not prompt-stuffing |
| 4 | Production scaffolding | Section 8 | Observability, backoff retries, rate limiting, typed errors, eval harness, unit + integration tests all present and runnable |
| 5 | Composable tool I/O | Section 4 (compose examples) | At least one tool consumes another tool's structured output |

## 4. Namespaces (MCP servers) and tools

Six servers. The enumeration below sums to 54 tools, deliberately above the hard floor of 50 so the build has margin (a judge closes Property 1 by counting registered tools, so do not land at exactly 50, and do not pad with filler: every tool below is real work the agent would call). Each tool has a pydantic input model and a pydantic output model. No tool returns a raw string where a structured object is meaningful.

**git** (version control, 12): status, diff, log, blame, branch_create, branch_list, checkout, commit, stash, show_commit, list_changed_files, tag.

**fs** (filesystem, 11): read_file, read_file_range, write_file, list_dir, search_files (glob), grep (content search), file_stat, make_dir, move, delete, copy.

**ast** (static analysis, 9): parse_module, list_symbols, find_definition, find_references, list_imports, compute_complexity, detect_dead_code, extract_function_signature, find_unused_imports.

**test** (test runner, 8): discover_tests, run_test_file, run_test_node, run_suite, coverage_report, coverage_diff, last_failures, rerun_failed.

**deps** (dependencies, 7): list_dependencies, check_outdated, resolve_import, find_unused_deps, dependency_graph, vulnerability_scan, add_dependency.

**ci** (continuous integration / quality, 7): run_linter, run_formatter, run_type_check, build_check, pre_commit_run, run_security_scan, summarize_quality.

Count: git 12 + fs 11 + ast 9 + test 8 + deps 7 + ci 7 = 54. If any namespace feels thin when you build it, cut it to fewer fat namespaces rather than padding, and recount to confirm you stay above 50.

Composition examples (Property 5), wired explicitly:
- `git.list_changed_files` output feeds `test.discover_tests` input, so the agent tests only what changed.
- `ast.find_references` output feeds `fs.read_file_range` to pull each call site.
- `test.last_failures` output feeds `fs.grep` to locate the failing assertions.

## 5. Tool-retrieval layer (the showpiece)

This is where the submission wins or blends in. Build it to a level the field will not reach.

Design:
1. At startup, the MCP client discovers all tools from all 6 servers and builds a registry: `{namespace, name, description, input_schema, output_schema}`.
2. Each tool's `description + namespace + signature` is embedded (all-MiniLM-L6-v2) and stored in pgvector.
3. On each agent step, the current goal/sub-goal is embedded and used to retrieve the top-k relevant tools (k tuned, start at 12) plus always-include core tools (git.status, fs.read_file).
4. Only the retrieved subset's schemas are passed to the model for that step. The model selects from the subset.
5. A retrieval-miss guard: if the model requests a tool not in the current subset, log the miss, widen k, and re-retrieve. This guard is itself an observability signal (retrieval quality metric).

Why this beats the alternative: dumping 50 schemas into every prompt is the lazy path and it degrades selection accuracy and burns tokens. Semantic retrieval keeps the prompt lean and the registry coherent at scale. This is the literal meaning of "remains coherent at fifty tools rather than collapsing into a chain of fifty conditional dispatches."

Eval this layer directly (Section 8): a labeled set of at least 40 to 60 (goal, correct-tool) pairs, measure retrieval recall@k. A set of ten proves nothing and will not survive questioning on the review call, so build the set out properly. This number goes in the MEMO.

## 6. Subagent contract

One parent-facing tool, `spawn_subagent(task: SubagentTask) -> SubagentResult`.

Isolation requirements (all enforced in code, not by convention):
- Fresh context: the subagent gets its own message history seeded only by the task brief, not the parent's full context.
- Scoped toolset: the task declares which namespaces the subagent may touch. Example: a "run and triage the test suite" subagent gets only `test` and `fs.read_file`, never `git.commit` or `fs.write_file`.
- Structured return: `SubagentResult` is a pydantic model (e.g. `status`, `findings: list[Finding]`, `artifacts`, `tokens_used`). The parent consumes the object, never the subagent's raw transcript.
- Its own budget: separate max-step and token budget so a runaway subagent cannot starve the parent.

Canonical subagent for this build: **test-triage subagent.** Parent makes a change, then spawns a subagent scoped to `test` + `fs.read_file` with the task "run the affected suite, identify failures, return structured findings." Subagent does its own multi-step loop in isolation and returns `SubagentResult`. Parent decides next action from the structured result.

This is a real spawn with its own loop and context. A helper function named `subagent()` that runs in the parent's context does not satisfy Property 2, and the brief says so explicitly.

## 7. Long-horizon execution and context management

A LangGraph node, `manage_context`, runs between steps and is the explicit strategy the brief wants visible.

Strategy:
- Track a running token estimate of the message history.
- When it crosses a threshold, compact: summarize completed sub-tasks into a compact "progress ledger" (what was done, what was decided, open threads), and drop the verbose intermediate tool outputs that are no longer needed, while preserving the plan and unresolved items verbatim.
- Keep a persistent `plan` object in graph state, separate from the message history, so plan coherence does not depend on the messages surviving compaction.

Demonstrable 20+ call task: "Add input validation to module X, update its callers, and make the suite green." This naturally spans status, search, read across several files, multiple writes, ast.find_references, repeated test runs, a subagent triage, and commits. Easily 20+ calls. The `plan` object plus the ledger keep coherence across the compaction boundary, which is the thing being graded.

## 8. Production scaffolding

Not optional, and the axis the field most often skips. This is where ISRO-grade discipline shows.

- **Observability:** LangSmith tracing on the graph, every node and tool call and subagent span. Structured logging (JSON) with correlation IDs per session and per subagent.
- **Retries with exponential backoff:** wrap all external calls (model API, MCP transport) with backoff + jitter and a max-attempts cap. Typed retry-exhausted error, not a bare raise.
- **Rate limiting:** a token-bucket limiter on outbound model and external calls. Configurable per-server.
- **Typed errors:** a small error hierarchy (`ToolError`, `RetryExhausted`, `SubagentBudgetExceeded`, `RetrievalMiss`, `ValidationError`). No bare exceptions crossing module boundaries. Tool failures return typed error results the model can reason about, not stack traces.
- **Eval harness:** two evals. (a) Retrieval recall@k on a labeled set of at least 40 to 60 (goal, tool) pairs (not ten, a small set is theater). (b) End-to-end task success on a small fixture repo with seeded bugs: does the agent make the suite green within budget. Both runnable via one command, both emit metrics.
- **Tests:** unit (tool input/output models, retrieval ranking, context compaction logic, backoff behavior, subagent isolation) and integration (a full agent run against a fixture repo, a full subagent spawn-and-return path). Both paths covered.
- **Deployable layout:** structured package (Section 10), config via environment, no notebook, a Dockerfile, a Makefile with `make test`, `make eval`, `make run`.

## 9. Build order (vertical slices, one commit story per slice)

Build depth-first down a thin line, then widen. Each slice is a real commit with a real message. Do not commit one giant dump. The commit history is read.

1. **Skeleton + one tool end to end.** LangGraph loop, MCP client, `fs.read_file` only, typed I/O, one passing unit test. Proves the spine.
2. **git + fs servers full, ~20 tools.** Plus the first integration test against a fixture repo.
3. **Retrieval layer.** pgvector, embeddings, top-k selection, retrieval-miss guard, recall@k eval. This is the showpiece, give it real time.
4. **Remaining servers (ast, test, deps, ci) to cross 50 tools.** Add composition wiring (Section 4 examples).
5. **Subagent.** Contract, isolation, scoped toolset, structured return, test-triage subagent, isolation unit test + spawn integration test.
6. **Context manager.** Compaction, persistent plan object, the 20+ call demo task.
7. **Scaffolding hardening.** Backoff, rate limiting, error hierarchy, LangSmith wiring, end-to-end eval.
8. **MEMO + video + trace export.** Write MEMO from this spec's decisions. Record walkthrough. Export session JSONL.

If day 3 ends and slices 1 to 4 are shaky, this is the point to consider the deep-research fallback. The retrieval, MCP, and LangSmith pieces transfer unchanged. Decide at day 3, not day 5.

## 10. Repository layout

```
.
├── MEMO.md
├── SPEC.md
├── Dockerfile
├── Makefile
├── pyproject.toml
├── README.md
├── src/
│   └── agent/
│       ├── graph/            # LangGraph nodes: plan, retrieve, act, manage_context
│       ├── mcp_client/       # discovery, registry, transport
│       ├── servers/          # the 6 MCP servers (git, fs, ast, test, deps, ci)
│       ├── retrieval/        # embeddings, pgvector store, top-k, miss guard
│       ├── subagent/         # contract, runner, isolation
│       ├── reliability/      # backoff, rate limit, error hierarchy
│       ├── observability/    # logging, LangSmith setup, correlation IDs
│       └── models/           # shared pydantic models (tool I/O, results, errors)
├── evals/
│   ├── retrieval/            # labeled (goal, tool) pairs, recall@k
│   └── e2e/                  # fixture repo with seeded bugs, task success
└── tests/
    ├── unit/
    └── integration/
```

## 11. MEMO seeds (fill as you build, do not invent at the end)

- **What I built:** one or two paragraphs, the harness and the five properties.
- **What I cut:** be specific and honest (e.g. cut a 7th namespace, cut multi-subagent parallelism, capped eval set size). Honest cuts read as judgment, not as gaps.
- **What more time would address:** the named cuts plus the next thing (e.g. parallel subagents, larger labeled eval set, streaming).
- **One design decision I would defend:** the tool-retrieval layer over a flat 50-schema prompt. State the alternative an engineer might reasonably pick (just pass all 50, models have big context now), then defend retrieval on selection accuracy, token cost, and the explicit meaning of "coherent at fifty." This is the strongest defensible decision in the build.

## 12. Divergence moment (for the video, capture it live, do not stage it)

Likely real moment: Claude Code will, at some step, propose collapsing the retrieval layer into a simpler flat dispatch or a hand-written router because it is faster to write. That is the exact thing Property 1 forbids. Overrule it, in the trace, with the reason. That disagreement is genuine, it will happen, and it is the perfect "moment where you and the model diverged" the brief asks for. Let it happen, do not pre-empt it, and keep the trace.

## 13. How Claude Code is used here (so the trace reads right)

The model writes most of the code. The judgment is yours: every architectural call in this spec, each slice's review, the pushback when output drifts from the spec, and the captured divergence. Build slice by slice (Section 9), read what it writes before committing, and keep the session JSONL flowing the whole time. The trace is the work sample. Treat it as one.
