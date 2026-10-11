from oracle.anomaly import AnomalyDetector


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
