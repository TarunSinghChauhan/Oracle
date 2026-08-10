import os
import tempfile
import pytest

from oracle.tracer import Tracer
from oracle.storage import TraceStore


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = TraceStore(path)
    yield s
    s.close()
    os.remove(path)

def test_save_and_retrieve_span(store):
    tracer = Tracer(store=store)
    with tracer.span("op", kind="tool_call"):
        pass
    rows = store.get_spans()
    assert len(rows) == 1
    assert rows[0]["name"] == "op"
    assert rows[0]["kind"] == "tool_call"

def test_filter_by_kind(store):
    tracer = Tracer(store=store)
    with tracer.span("a", kind="llm_call"):
        pass
    with tracer.span("b", kind="tool_call"):
        pass
    rows = store.get_spans(kind="llm_call")
    assert len(rows) == 1
    assert rows[0]["name"] == "a"

def test_filter_by_trace_id(store):
    t1 = Tracer(store=store)
    t2 = Tracer(store=store)
    with t1.span("a"):
        pass
    with t2.span("b"):
        pass
    rows = store.get_spans(trace_id=t1.trace_id)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == t1.trace_id

def test_attributes_round_trip_as_dict(store):
    tracer = Tracer(store=store)
    with tracer.span("op", attributes={"cost_usd": 0.002}):
        pass
    rows = store.get_spans()
    assert rows[0]["attributes"]["cost_usd"] == 0.002

def test_limit_respected(store):
    tracer = Tracer(store=store)
    for i in range(5):
        with tracer.span(f"op-{i}"):
            pass
    rows = store.get_spans(limit=3)
    assert len(rows) == 3
