import os
import tempfile
import threading
import time
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


def _make_span(i: int):
    from oracle.tracer import Span

    return Span(
        span_id=f"span-{i}",
        name=f"op-{i}",
        start_time=time.time(),
        end_time=time.time(),
        trace_id="t",
        kind="tool_call",
    )


def test_save_span_from_another_thread(store):
    errors = []

    def worker():
        try:
            store.save_span(_make_span(0))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert errors == []
    assert [r["span_id"] for r in store.get_spans()] == ["span-0"]


def test_concurrent_writers_lose_no_spans(store):
    n_threads, per_thread = 8, 25
    errors = []

    def worker(offset: int):
        try:
            for i in range(per_thread):
                store.save_span(_make_span(offset * per_thread + i))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store.get_spans(limit=10_000)) == n_threads * per_thread


def test_reads_and_writes_can_overlap_across_threads(store):
    errors = []

    def writer():
        try:
            for i in range(50):
                store.save_span(_make_span(i))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def reader():
        try:
            for _ in range(50):
                store.get_spans()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
