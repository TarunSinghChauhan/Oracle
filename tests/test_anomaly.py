import os
import tempfile

import pytest

from oracle.anomaly import AnomalyDetector
from oracle.storage import TraceStore
from oracle.tracer import Span


def _mk_span(span_id, name="llm_call", duration_ms=100.0, cost=0.005, status="ok"):
    return {
        "span_id": span_id,
        "name": name,
        "duration_ms": duration_ms,
        "attributes": {"cost_usd": cost},
        "status": status,
    }


def test_no_anomaly_with_insufficient_history():
    det = AnomalyDetector(min_samples=5)
    history = [_mk_span(f"h{i}") for i in range(2)]
    span = _mk_span("s1", duration_ms=100)
    assert det.check_latency(span, history) is None

def test_latency_spike_detected():
    det = AnomalyDetector(min_samples=5, latency_z_threshold=3.0)
    history = [_mk_span(f"h{i}", duration_ms=100.0) for i in range(10)]
    spike = _mk_span("s1", duration_ms=5000.0)
    result = det.check_latency(spike, history)
    assert result is not None
    assert result.kind == "latency_spike"

def test_no_latency_spike_within_normal_range():
    det = AnomalyDetector(min_samples=5)
    history = [_mk_span(f"h{i}", duration_ms=100.0 + i) for i in range(10)]
    normal = _mk_span("s1", duration_ms=105.0)
    assert det.check_latency(normal, history) is None

def test_cost_spike_detected():
    det = AnomalyDetector(min_samples=5, cost_z_threshold=3.0)
    history = [_mk_span(f"h{i}", cost=0.005) for i in range(10)]
    spike = _mk_span("s1", cost=1.0)
    result = det.check_cost(spike, history)
    assert result is not None
    assert result.kind == "cost_spike"

def test_cost_check_ignores_spans_without_cost():
    det = AnomalyDetector()
    span = {"span_id": "s1", "name": "op", "attributes": {}}
    assert det.check_cost(span, []) is None

def test_error_burst_detected():
    det = AnomalyDetector(min_samples=5)
    spans = [_mk_span(f"e{i}", status="error") for i in range(8)] + \
            [_mk_span(f"o{i}", status="ok") for i in range(2)]
    result = det.check_error_burst(spans, window=10, rate_threshold=0.3)
    assert result is not None
    assert result.kind == "error_burst"

def test_no_error_burst_below_threshold():
    det = AnomalyDetector(min_samples=5)
    spans = [_mk_span(f"o{i}", status="ok") for i in range(10)]
    result = det.check_error_burst(spans, window=10, rate_threshold=0.3)
    assert result is None

def test_zero_stdev_returns_zero_zscore():
    det = AnomalyDetector(min_samples=3)
    history = [_mk_span(f"h{i}", duration_ms=100.0) for i in range(5)]
    same = _mk_span("s1", duration_ms=100.0)
    assert det.check_latency(same, history) is None


def test_zero_variance_baseline_flags_slower_span():
    det = AnomalyDetector(min_samples=3)
    history = [_mk_span(f"h{i}", duration_ms=100.0) for i in range(5)]
    slow = _mk_span("s1", duration_ms=500.0)
    result = det.check_latency(slow, history)
    assert result is not None
    assert result.kind == "latency_spike"
    assert result.severity == "critical"


def test_zero_variance_baseline_ignores_faster_span():
    det = AnomalyDetector(min_samples=3)
    history = [_mk_span(f"h{i}", duration_ms=100.0) for i in range(5)]
    fast = _mk_span("s1", duration_ms=10.0)
    assert det.check_latency(fast, history) is None


def test_zero_variance_baseline_flags_costlier_span():
    det = AnomalyDetector(min_samples=3)
    history = [_mk_span(f"h{i}", cost=0.01) for i in range(5)]
    pricey = _mk_span("s1", cost=0.5)
    result = det.check_cost(pricey, history)
    assert result is not None
    assert result.kind == "cost_spike"


def test_zero_variance_baseline_ignores_cheaper_span():
    det = AnomalyDetector(min_samples=3)
    history = [_mk_span(f"h{i}", cost=0.01) for i in range(5)]
    cheap = _mk_span("s1", cost=0.0)
    assert det.check_cost(cheap, history) is None


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = TraceStore(path)
    yield s
    s.close()
    os.remove(path)


def _save(store, i, duration_ms=100.0, kind="llm_call", name="llm_call",
          status="ok", cost=0.005):
    """Save span number `i`; higher `i` means a later start_time (newer)."""
    start = 1_000.0 + i
    store.save_span(Span(
        span_id=f"s{i}",
        name=name,
        start_time=start,
        end_time=start + duration_ms / 1000.0,
        trace_id="t",
        kind=kind,
        status=status,
        attributes={"cost_usd": cost},
    ))


def _seed_baseline(store, n=10, offset=0, **kw):
    # Small, non-zero spread so the baseline has a real standard deviation.
    for i in range(n):
        _save(store, offset + i, duration_ms=100.0 + i, cost=0.005 + i * 0.0001, **kw)


def test_scan_returns_nothing_for_healthy_store(store):
    _seed_baseline(store)
    assert AnomalyDetector().scan(store) == []


def test_scan_returns_nothing_for_empty_store(store):
    assert AnomalyDetector().scan(store) == []


def test_scan_flags_latency_spike_in_store(store):
    _seed_baseline(store)
    _save(store, 10, duration_ms=5000.0)

    found = AnomalyDetector().scan(store)

    assert [(a.kind, a.span_id) for a in found] == [("latency_spike", "s10")]
    assert found[0].value == pytest.approx(5000.0)


def test_scan_flags_cost_spike_in_store(store):
    _seed_baseline(store)
    _save(store, 10, cost=2.0)

    found = AnomalyDetector().scan(store)

    assert [(a.kind, a.span_id) for a in found] == [("cost_spike", "s10")]


def test_scan_reports_error_burst(store):
    for i in range(10):
        _save(store, i, status="error" if i >= 3 else "ok")

    found = AnomalyDetector().scan(store)

    bursts = [a for a in found if a.kind == "error_burst"]
    assert len(bursts) == 1
    assert bursts[0].severity == "critical"
    assert bursts[0].value == pytest.approx(0.7)


def test_scan_kind_filter_limits_what_is_checked(store):
    _seed_baseline(store, kind="tool_call", name="search")
    _save(store, 10, duration_ms=5000.0, kind="tool_call", name="search")
    _seed_baseline(store, offset=100, kind="llm_call", name="llm_call")
    det = AnomalyDetector()

    assert any(a.kind == "latency_spike" for a in det.scan(store, kind="tool_call"))
    assert det.scan(store, kind="llm_call") == []


def test_scan_does_not_judge_a_span_against_newer_history(store):
    # The slow span is the *oldest*, so it has no prior history to be compared
    # with and must not be flagged by a baseline built from later spans.
    _save(store, 0, duration_ms=5000.0)
    for i in range(1, 11):
        _save(store, i, duration_ms=100.0 + i)

    assert AnomalyDetector().scan(store) == []
