"""Isolated subagent loop: scoped tools, fresh context, typed result, own budget."""
from __future__ import annotations

import json
from typing import Any

import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from agent.subagent.contract import (
    Finding,
    NamespaceScope,
    SubagentBudget,
    SubagentBudgetExceeded,
    SubagentResult,
    SubagentTask,
    ToolScopeViolation,
)

_SYSTEM = SystemMessage(content=(
    "You are a focused test-triage agent. Use only the tools available to you. "
    "Run the requested tests, identify failures, then stop calling tools. "
    "Do not attempt to fix anything; only report what you find."
))

_TEST_RUNNER_TOOLS = frozenset({
    "run_suite", "run_test_file", "run_test_node", "rerun_failed",
    "discover_tests", "coverage_report", "coverage_diff", "last_failures",
})


def _parse_raw(raw: Any) -> dict:
    """Normalise a tool's arun() return to a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"output": raw}
        except json.JSONDecodeError:
            # MCP sometimes wraps the result as a list with a text block.
            pass
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict) and first.get("type") == "text":
            try:
                parsed = json.loads(first["text"])
                return parsed if isinstance(parsed, dict) else {"output": first["text"]}
            except (json.JSONDecodeError, KeyError):
                return {"output": str(first.get("text", raw))}
    return {"output": str(raw)}


def _extract_findings(tool_results: list[tuple[str, dict]]) -> list[Finding]:
    """Build Finding objects from collected test tool outputs."""
    findings: list[Finding] = []
    seen: set[str] = set()

    for tool_name, result in tool_results:
        # run_suite / run_test_file / run_test_node / rerun_failed all carry a
        # "results" list of TestResult-shaped dicts.
        if tool_name in {"run_suite", "run_test_file", "rerun_failed"}:
            for r in result.get("results", []):
                outcome = r.get("outcome", "")
                if outcome == "passed":
                    continue
                nid = r.get("node_id", "unknown")
                if nid in seen:
                    continue
                seen.add(nid)
                findings.append(Finding(
                    test_id=nid,
                    status=outcome if outcome in ("failed", "error", "skipped") else "error",
                    message=(r.get("longrepr") or "")[:500],
                ))

        elif tool_name == "run_test_node":
            outcome = result.get("outcome", "")
            if outcome != "passed":
                nid = result.get("node_id", "unknown")
                if nid not in seen:
                    seen.add(nid)
                    findings.append(Finding(
                        test_id=nid,
                        status=outcome if outcome in ("failed", "error", "skipped") else "error",
                        message=(result.get("longrepr") or "")[:500],
                    ))

        elif tool_name == "last_failures":
            for f in result.get("failures", []):
                tid = f.get("test_id", "unknown")
                if tid in seen:
                    continue
                seen.add(tid)
                findings.append(Finding(
                    test_id=tid,
                    status="failed",
                    message=(f.get("error_text") or "")[:500],
                    file_path=f.get("file"),
                    line=f.get("line"),
                ))

    return findings


def _extract_artifacts(tool_results: list[tuple[str, dict]]) -> dict[str, str]:
    """Store raw test tool outputs as string artifacts."""
    artifacts: dict[str, str] = {}
    for tool_name, result in tool_results:
        if tool_name in _TEST_RUNNER_TOOLS:
            key = f"{tool_name}_output"
            artifacts[key] = json.dumps(result)[:2000]
    return artifacts


class SubagentRunner:
    """Runs an isolated agent loop with a scoped toolset and its own budget.

    The subagent gets a fresh message history seeded only by the task brief.
    It cannot call tools outside the declared NamespaceScope list because those
    tool objects are never passed to its LLM binding.
    """

    def __init__(self, all_tools_by_namespace: dict[str, list]) -> None:
        self._all_tools_by_namespace = all_tools_by_namespace

    def _scope_tools(self, scopes: list[NamespaceScope]) -> list:
        """Return the filtered tool list for the given scopes.

        Raises ValueError if a namespace or named tool is not available.
        """
        result: list = []
        for scope in scopes:
            ns_tools = self._all_tools_by_namespace.get(scope.namespace)
            if ns_tools is None:
                raise ValueError(
                    f"unknown namespace {scope.namespace!r}; "
                    f"available: {sorted(self._all_tools_by_namespace)}"
                )
            if scope.tools is None:
                result.extend(ns_tools)
            else:
                by_name = {t.name: t for t in ns_tools}
                for name in scope.tools:
                    if name not in by_name:
                        raise ValueError(
                            f"tool {name!r} not found in namespace {scope.namespace!r}; "
                            f"available: {sorted(by_name)}"
                        )
                    result.append(by_name[name])
        return result

    async def run(self, task: SubagentTask) -> SubagentResult:
        """Execute the subagent loop and return a typed result.

        Isolation guarantees enforced here:
        - messages starts fresh (task brief only, no parent history)
        - LLM is bound only to scoped_tools
        - Every tool call goes through tool_map, which contains only scoped tools
        - Budget is checked at the start of each iteration before any LLM call
        """
        scoped_tools = self._scope_tools(task.allowed_scopes)
        tool_map = {t.name: t for t in scoped_tools}

        llm = ChatGroq(model=os.environ.get("AGENT_MODEL", "llama-3.1-8b-instant"))
        bound = llm.bind_tools(scoped_tools)

        messages: list = [_SYSTEM, HumanMessage(content=task.brief)]
        steps = 0
        tokens_used = 0
        tool_results: list[tuple[str, dict]] = []

        while True:
            # Budget check before every LLM call so an over-budget step never runs.
            if steps >= task.budget.max_steps:
                raise SubagentBudgetExceeded(steps, tokens_used, task.budget)
            if tokens_used >= task.budget.max_tokens:
                raise SubagentBudgetExceeded(steps, tokens_used, task.budget)

            print(
                f"[subagent] step {steps + 1}/{task.budget.max_steps}, "
                f"tokens {tokens_used}/{task.budget.max_tokens}",
                flush=True,
            )
            response = await bound.ainvoke(messages)
            steps += 1
            tokens_used += (response.usage_metadata or {}).get("total_tokens", 0)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool = tool_map.get(tc["name"])
                if tool is None:
                    # Double-enforcement: the model should only have seen scoped
                    # schemas, but catch a violation explicitly rather than letting
                    # it silently fall through to an unrelated tool.
                    raise ToolScopeViolation(tc["name"], list(tool_map))

                print(f"[subagent] tool {tc['name']}  args={str(tc['args'])[:80]}", flush=True)
                raw = await tool.arun(tc["args"])
                result_dict = _parse_raw(raw)
                tool_results.append((tc["name"], result_dict))
                messages.append(
                    ToolMessage(content=str(raw), tool_call_id=tc["id"])
                )

        summary = ""
        if hasattr(response, "content") and isinstance(response.content, str):
            summary = response.content

        print(
            f"[subagent] done: {steps} steps, {tokens_used} tokens, "
            f"{len(tool_results)} tool calls",
            flush=True,
        )

        return SubagentResult(
            status="completed",
            findings=_extract_findings(tool_results),
            artifacts=_extract_artifacts(tool_results),
            tokens_used=tokens_used,
            steps_taken=steps,
            summary=summary,
        )
