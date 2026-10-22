import os
import tempfile

import pytest

from oracle.scan import scan_all
from oracle.storage import TraceStore
from oracle.tracer import Span

STEADY = "the quarterly report shows steady revenue growth across all regions this year"
BROKEN = "error upstream service unavailable please retry the request later today"


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = TraceStore(path)
    yield s
    s.close()
    os.remove(path)


def _save(store, i, duration_ms=100.0, output=None, kind="llm_call"):
    start = 1_000.0 + i
    store.save_span(Span(
        span_id=f"s{i}", name="summarize", start_time=start,
        end_time=start + duration_ms / 1000.0, trace_id="t", kind=kind,
        attributes={} if output is None else {"output": output},
    ))


def test_scan_all_on_empty_store_is_empty(store):
    assert scan_all(store) == []


def test_scan_all_reports_drift_alongside_other_anomalies(store):
    for i in range(10):
        _save(store, i, duration_ms=100.0 + i, output=STEADY)
    _save(store, 10, duration_ms=5000.0, output=BROKEN)

    kinds = sorted(a.kind for a in scan_all(store))

    assert kinds == ["drift", "latency_spike"]


def test_scan_all_is_quiet_for_a_healthy_store(store):
    for i in range(10):
        _save(store, i, duration_ms=100.0 + i, output=STEADY)
    assert scan_all(store) == []


def test_scan_all_passes_the_kind_filter_to_both_detectors(store):
    for i in range(10):
        _save(store, i, duration_ms=100.0 + i, output=STEADY, kind="llm_call")
    _save(store, 10, duration_ms=5000.0, output=BROKEN, kind="llm_call")

    assert scan_all(store, kind="tool_call") == []
    assert len(scan_all(store, kind="llm_call")) == 2
