"""Structured JSON logging for the agent.

Each log line is a single JSON object with fields:
  ts, level, logger, msg, correlation_id, + any extra kwargs passed by the caller.

Usage:
    from agent.logging_config import configure_logging, get_logger
    configure_logging()   # call once at startup
    log = get_logger(__name__)
    log.info("tool called", extra={"correlation_id": cid, "tool": "read_file"})

LangSmith tracing is enabled automatically by LangChain when both env vars
are present; no code changes needed:
    LANGCHAIN_TRACING_V2=true
    LANGSMITH_API_KEY=<your-key>
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = getattr(record, "correlation_id", None)
        if cid:
            doc["correlation_id"] = cid
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "correlation_id", "taskName",
            }:
                try:
                    json.dumps(val)
                    doc[key] = val
                except (TypeError, ValueError):
                    doc[key] = str(val)
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


def configure_logging(level: str | None = None) -> None:
    """Install the JSON handler on the root logger. Call once at startup."""
    lvl = getattr(logging, (level or os.environ.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
