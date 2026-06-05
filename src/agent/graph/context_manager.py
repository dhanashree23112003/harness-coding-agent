"""Slice 6: manage_context node - token tracking and message compaction.

Strategy (SPEC Section 7):
- Estimate tokens in the message history after every tool execution.
- When the estimate crosses COMPACT_THRESHOLD, compact: summarize completed
  tool-call pairs into a progress ledger and drop their verbose messages.
- The `plan` and `progress_ledger` fields in AgentState are separate from
  messages, so plan coherence survives compaction without prompt-stuffing.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, SystemMessage, ToolMessage

from agent.graph.state import AgentState

COMPACT_THRESHOLD: int = int(os.environ.get("CONTEXT_COMPACT_THRESHOLD", "1500"))
KEEP_RECENT_PAIRS: int = int(os.environ.get("CONTEXT_KEEP_PAIRS", "3"))


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Estimate token count as character count / 4 (conservative, no tokenizer needed)."""
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", ""))
                else:
                    total += len(str(block))
    return total // 4


def _extract_pairs(
    messages: list[BaseMessage],
) -> list[tuple[AIMessage, list[ToolMessage]]]:
    """Group AIMessage-with-tool-calls with their following ToolMessages into pairs.

    Skips SystemMessages, HumanMessages, and AIMessages without tool_calls.
    Pairs are returned in message order.
    """
    pairs: list[tuple[AIMessage, list[ToolMessage]]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_msgs: list[ToolMessage] = []
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tool_msgs.append(messages[j])
                j += 1
            if tool_msgs:
                pairs.append((msg, tool_msgs))
                i = j
            else:
                i += 1
        else:
            i += 1
    return pairs


def _summarize_pair(ai_msg: AIMessage, tool_msgs: list[ToolMessage]) -> str:
    """Build a deterministic one-line summary for a completed tool-call pair.

    No LLM call. Extracts key fields from ToolMessage JSON where possible.
    """
    lines: list[str] = []
    tool_names = [tc["name"] for tc in (ai_msg.tool_calls or [])]

    for i, tmsg in enumerate(tool_msgs):
        name = tool_names[i] if i < len(tool_names) else "unknown"
        raw = tmsg.content
        if isinstance(raw, list) and raw:
            first = raw[0]
            raw = first.get("text", "") if isinstance(first, dict) else str(first)

        detail = ""
        try:
            parsed: Any = json.loads(raw) if isinstance(raw, str) else {}
            if isinstance(parsed, dict):
                if "path" in parsed:
                    detail = f"path={parsed['path']}"
                elif "summary" in parsed:
                    detail = str(parsed["summary"])[:80]
                elif "passed" in parsed or "failed" in parsed:
                    p = parsed.get("passed", 0)
                    f = parsed.get("failed", 0)
                    detail = f"passed={p} failed={f}"
                elif "output" in parsed:
                    detail = str(parsed["output"])[:80]
                else:
                    detail = str(raw)[:80]
            else:
                detail = str(raw)[:80]
        except (json.JSONDecodeError, TypeError):
            detail = str(raw)[:80]

        lines.append(f"  {name}({detail})")

    return "\n".join(lines)


def build_ledger(
    pairs_to_compact: list[tuple[AIMessage, list[ToolMessage]]],
    prior_ledger: str,
) -> str:
    """Build the full progress ledger string from prior ledger + new summaries."""
    new_entries = [_summarize_pair(ai, tools) for ai, tools in pairs_to_compact]
    combined = []
    if prior_ledger:
        combined.append(prior_ledger)
    combined.extend(new_entries)
    return "Progress ledger (completed steps):\n" + "\n".join(combined)


def compact_messages(state: AgentState) -> dict:
    """Compact the message history when the token estimate exceeds the threshold.

    Keeps the last KEEP_RECENT_PAIRS tool-call pairs intact. Older pairs are
    summarized into the progress ledger and their messages are removed.
    The plan field in state is untouched (it is not a message).
    """
    messages = state.get("messages", [])
    all_pairs = _extract_pairs(messages)

    if len(all_pairs) <= KEEP_RECENT_PAIRS:
        # Not enough pairs to compact; just update the estimate.
        return {"token_estimate": estimate_tokens(messages)}

    pairs_to_compact = all_pairs[:-KEEP_RECENT_PAIRS]
    new_ledger = build_ledger(pairs_to_compact, state.get("progress_ledger", ""))

    # Collect IDs to remove.
    remove_ids: set[str] = set()
    for ai_msg, tool_msgs in pairs_to_compact:
        remove_ids.add(ai_msg.id)
        for tm in tool_msgs:
            remove_ids.add(tm.id)

    # Also remove prior ledger message if it exists.
    prior_ledger_id = state.get("ledger_message_id")
    if prior_ledger_id:
        remove_ids.add(prior_ledger_id)

    remove_ops = [RemoveMessage(id=rid) for rid in remove_ids if rid]

    # Inject a new SystemMessage carrying the ledger so the model can read it.
    ledger_msg = SystemMessage(content=new_ledger, id=str(uuid.uuid4()))

    compaction_count = state.get("compaction_count", 0) + 1
    remaining_messages = [m for m in messages if m.id not in remove_ids]
    new_estimate = estimate_tokens(remaining_messages) + estimate_tokens([ledger_msg])

    stale_tool_count = sum(len(tms) for _, tms in pairs_to_compact)
    print(
        f"[compaction] tokens {state.get('token_estimate', estimate_tokens(messages))} "
        f"over threshold {COMPACT_THRESHOLD}, built ledger, "
        f"dropped {stale_tool_count} stale tool messages, plan preserved",
        flush=True,
    )

    return {
        "messages": remove_ops + [ledger_msg],
        "progress_ledger": new_ledger,
        "token_estimate": new_estimate,
        "compaction_count": compaction_count,
        "ledger_message_id": ledger_msg.id,
    }


def manage_context_node(state: AgentState) -> dict:
    """LangGraph node: runs after every tool execution to check context size.

    If the token estimate is below COMPACT_THRESHOLD, only updates the estimate.
    If at or above the threshold, compacts the message history.
    """
    messages = state.get("messages", [])
    estimate = estimate_tokens(messages)

    if estimate < COMPACT_THRESHOLD:
        return {"token_estimate": estimate}

    return compact_messages(state)
