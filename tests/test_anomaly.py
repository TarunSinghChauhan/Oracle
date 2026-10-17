import os
import random
import statistics
import tempfile
import time

import pytest

from oracle.anomaly import AnomalyDetector, _RunningStats
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


def test_scan_matches_checking_each_span_against_all_older_history(store):
    # scan() keeps running per-name statistics instead of re-reading history for
    # every span. It must report exactly what the per-span check_* methods
    # report when each span is compared against every older span.
    rng = random.Random(7)
    for i in range(240):
        name = rng.choice(["llm_call", "tool_call"])
        if 100 <= i < 140:
            duration, cost = 100.0, 0.005  # flat stretch: zero-variance baselines
        else:
            duration = rng.choice([100.0, 105.0, 110.0, 400.0])
            cost = rng.choice([0.004, 0.005, 0.006, 0.09, None])
        start = 1_000.0 + i
        store.save_span(Span(
            span_id=f"s{i}", name=name, start_time=start,
            end_time=start + duration / 1000.0, trace_id="t", kind="llm_call",
            attributes={} if cost is None else {"cost_usd": cost},
        ))

    det = AnomalyDetector()
    spans = store.get_spans(limit=2000)  # newest first
    expected = []
    for i, span in enumerate(spans):
        for check in (det.check_latency, det.check_cost):
            found = check(span, spans[i + 1:])
            if found:
                expected.append(found)
    burst = det.check_error_burst(spans)
    if burst:
        expected.append(burst)

    assert expected, "fixture should produce some anomalies"
    assert det.scan(store) == expected


def test_scan_skips_unfinished_spans(store):
    _seed_baseline(store)
    store.save_span(Span(span_id="open", name="llm_call", start_time=2_000.0,
                         trace_id="t", kind="llm_call"))  # never ended

    assert AnomalyDetector().scan(store) == []
    unfinished = store.get_spans(limit=1)[0]
    assert unfinished["duration_ms"] is None
    assert AnomalyDetector().check_latency(unfinished, store.get_spans()) is None


def test_scan_cost_is_linear_in_span_count():
    # Re-reading all older history for every span made scan() quadratic: about
    # 2.4s for 2000 spans, and the dashboard rescans every few seconds. The
    # linear version takes a few hundredths of a second.
    rng = random.Random(1)

    class ListStore:  # scan() only needs get_spans(); skip SQLite for speed
        def __init__(self, spans):
            self.spans = spans

        def get_spans(self, kind=None, limit=1000):
            return self.spans[:limit]

    spans = [
        {
            "span_id": f"s{i}",
            "name": rng.choice(["a", "b", "c"]),
            "kind": "llm_call",
            "status": "ok",
            "duration_ms": rng.uniform(80, 120),
            "attributes": {"cost_usd": rng.uniform(0.004, 0.006)},
        }
        for i in range(2000)
    ]  # newest first, like TraceStore.get_spans

    started = time.perf_counter()
    AnomalyDetector().scan(ListStore(spans))
    assert time.perf_counter() - started < 1.0


def test_running_stats_match_statistics_module():
    rng = random.Random(3)
    values = [rng.uniform(0, 500) for _ in range(200)]
    stats = _RunningStats()
    for i, v in enumerate(values, start=1):
        stats.add(v)
        assert stats.n == i
        assert stats.mean == pytest.approx(statistics.mean(values[:i]), rel=1e-12)
        assert stats.stdev == pytest.approx(statistics.pstdev(values[:i]), rel=1e-9, abs=1e-12)


def test_running_stats_identical_values_have_exactly_zero_stdev():
    stats = _RunningStats()
    for _ in range(50):
        stats.add(0.1)  # not exactly representable; float running sums drift here
    assert stats.stdev == 0.0
    assert stats.mean == 0.1
