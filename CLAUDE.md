# CLAUDE.md

Operating instructions for Claude Code on this repository. Read this first, then read SPEC.md. SPEC.md is the source of truth for architecture. This file is the source of truth for how we work.

## What this repo is

A production-shaped autonomous coding agent built for a hiring hackathon judged by X-Arc, an applied AI lab that builds the engineering layer between frontier models and the systems around them. They grade the harness (tool routing, context management, subagent isolation, reliability), not domain cleverness. Five required properties are listed in SPEC.md Section 3. The build plan is SPEC.md Section 9.

## Who I am (so your defaults match mine)

Final-year AI/ML engineer, ex-ISRO ML intern. I value production discipline, typed code, honest reporting of tradeoffs, and measurable decisions over surface familiarity. Give me blunt, specific reasoning. No filler. Never use em dashes in anything you write (code comments, commits, docs): use commas, periods, parentheses, or colons.

## How we build (non-negotiable)

The commit history and the session traces are part of the graded submission. The way this gets built is itself the work sample. So:

1. **Vertical slices, one at a time.** Build only the current slice from SPEC.md Section 9. Do not build ahead. Do not scaffold future slices "to save time."
2. **Explain before you write.** Before writing code for a slice, state the file layout and approach in a sentence or two so I can catch drift early.
3. **I read before we commit.** You implement, I review against the spec, I push back or accept, then we commit. Surface any corner you are tempted to cut.
4. **One real commit per meaningful step,** with a real message (e.g. `slice 1: agent loop + fs.read_file end to end`). Never one giant dump. Small honest commits beat few big ones.
5. **Stop at the slice boundary.** When the current slice runs end to end, stop and say so. Do not roll into the next slice.

## Hard rules pulled from the spec (do not violate)

- No hand-routed tool dispatch. Tool selection is model-driven via the retrieval layer (SPEC Section 5). If you are ever tempted to write an if/else chain that picks tools, stop: that defeats Property 1 and is the exact failure the brief hunts for.
- The subagent (SPEC Section 6) is a real spawn with its own context, scoped toolset, and a typed structured return. A helper function named `subagent` in the parent context does not count.
- The context-management strategy (SPEC Section 7) lives in explicit code, not in prompt stuffing.
- Tool I/O is typed (pydantic in, pydantic out). No raw strings where a structured object is meaningful.

## When we disagree

If you propose a shortcut that conflicts with the spec, say so plainly and make your case. I may overrule you. That disagreement is expected and is captured on purpose. Do not silently comply, and do not silently cut corners.

## Stack

LangGraph (orchestration), MCP servers per namespace (tools), pgvector + sentence-transformer embeddings (tool retrieval), LangSmith (observability), Python 3.11+ typed throughout. Rationale in SPEC Section 2.

## Current state

Slices 1-6 functionally complete (long-horizon task proven live). Next: Slice 7 scaffolding, then MEMO/video/submit.