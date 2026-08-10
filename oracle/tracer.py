"""
tracer.py — Span capture for agent/tool call runs.

A Span represents one unit of work (an LLM call, a tool invocation,
a full agent run). Spans nest via parent_id, so a full agent run can be
reconstructed as a tree. No external API calls happen here — this
module only records what your own code tells it happened.
"""

from __future__ import annotations

import time
import uuid
import contextlib
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


@dataclass
class Span:
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    trace_id: str = ""
    kind: str = "generic"          # e.g. "llm_call", "tool_call", "agent_run"
    status: str = "ok"             # "ok" | "error"
    attributes: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_ms"] = self.duration_ms
        return d


class Tracer:
    """
    Records spans in memory and flushes them to a TraceStore.

    Usage:
        tracer = Tracer(store)
        with tracer.span("agent_run", kind="agent_run") as run:
            with tracer.span("llm_call", kind="llm_call", parent=run,
                              attributes={"tokens": 512, "cost_usd": 0.003}):
                ...
    """

    def __init__(self, store=None, trace_id: Optional[str] = None):
        self.store = store
        self.trace_id = trace_id or str(uuid.uuid4())
        self._spans: list[Span] = []

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        kind: str = "generic",
        parent: Optional[Span] = None,
        attributes: Optional[dict] = None,
    ):
        s = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            start_time=time.time(),
            parent_id=parent.span_id if parent else None,
            trace_id=self.trace_id,
            kind=kind,
            attributes=attributes or {},
        )
        try:
            yield s
        except Exception as e:
            s.status = "error"
            s.error = str(e)
            raise
        finally:
            s.end_time = time.time()
            self._spans.append(s)
            if self.store is not None:
                self.store.save_span(s)

    def spans(self) -> list[Span]:
        return list(self._spans)
