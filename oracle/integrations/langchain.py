"""
integrations/langchain.py — LangChain/LangGraph callback adapter for ORACLE.

Wires LangChain's callback events (LLM calls, tool calls, chain/agent
runs) directly into oracle.Tracer, so any existing LangChain or
LangGraph pipeline gets full ORACLE tracing by just adding one callback
— no changes to your chain/agent logic required.

Requires `langchain-core` to be installed (optional dependency — ORACLE's
core modules never require it). If it isn't installed, importing this
module raises a clear ImportError rather than failing silently.

Usage:
    from oracle.storage import TraceStore
    from oracle.tracer import Tracer
    from oracle.integrations.langchain import OracleCallbackHandler

    store = TraceStore("oracle_traces.db")
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)

    chain.invoke({"input": "..."}, config={"callbacks": [handler]})
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError as e:
    raise ImportError(
        "OracleCallbackHandler requires langchain-core. "
        "Install it with: pip install langchain-core"
    ) from e

from oracle.tracer import Tracer, Span


class OracleCallbackHandler(BaseCallbackHandler):
    """
    Maps LangChain run_id lifecycle events onto ORACLE spans.

    LangChain identifies each unit of work (LLM call, tool call, chain
    step) with a `run_id`, and nests them via `parent_run_id`. This
    handler keeps a run_id -> ORACLE Span mapping so nesting carries
    through to the trace store correctly, matching how you'd nest spans
    manually with `tracer.span(..., parent=...)`.
    """

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self._open_spans: dict[UUID, Any] = {}  # run_id -> context manager
        self._spans: dict[UUID, Span] = {}       # run_id -> Span

    def _start(self, run_id: UUID, parent_run_id: Optional[UUID], name: str, kind: str, attributes: dict) -> None:
        parent_span = self._spans.get(parent_run_id) if parent_run_id else None
        cm = self.tracer.span(name, kind=kind, parent=parent_span, attributes=attributes)
        span = cm.__enter__()
        self._open_spans[run_id] = cm
        self._spans[run_id] = span

    def _end(self, run_id: UUID, error: Optional[BaseException] = None) -> None:
        cm = self._open_spans.pop(run_id, None)
        self._spans.pop(run_id, None)
        if cm is None:
            return
        if error is not None:
            try:
                cm.__exit__(type(error), error, error.__traceback__)
            except Exception:
                pass
        else:
            cm.__exit__(None, None, None)

    # --- LLM events -----------------------------------------------------

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        model = (serialized or {}).get("name", "unknown_model")
        self._start(
            run_id, parent_run_id, "llm_call", kind="llm_call",
            attributes={"model": model, "prompt_count": len(prompts)},
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        model = (serialized or {}).get("name", "unknown_model")
        self._start(
            run_id, parent_run_id, "llm_call", kind="llm_call",
            attributes={"model": model},
        )

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        span = self._spans.get(run_id)
        if span is not None:
            usage = getattr(response, "llm_output", None) or {}
            token_usage = usage.get("token_usage", {}) if isinstance(usage, dict) else {}
            if token_usage:
                span.attributes["tokens"] = token_usage.get("total_tokens")
        self._end(run_id)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id, error=error)

    # --- Tool events ------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        tool_name = (serialized or {}).get("name", "unknown_tool")
        self._start(
            run_id, parent_run_id, "tool_call", kind="tool_call",
            attributes={"tool": tool_name},
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id)

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id, error=error)

    # --- Chain / agent events ---------------------------------------------

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        # Recent langchain-core passes `serialized=None` for chains and delivers
        # the run name (a node name, or `run_name`) via the `name` kwarg.
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain_run"
        self._start(
            run_id, parent_run_id, name, kind="agent_run",
            attributes={},
        )

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id, error=error)
