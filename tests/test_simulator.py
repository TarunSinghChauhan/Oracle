import random
from unittest.mock import patch

import pytest

from oracle.drift import DriftDetector, jaccard_similarity
from oracle.simulator import (
    DRIFTED_OUTPUTS,
    LATENCY_SCALE,
    NORMAL_OUTPUT,
    _simulate_llm_call,
    generate_traces,
)
from oracle.storage import TraceStore
from oracle.tracer import Tracer


@pytest.fixture
def store():
    s = TraceStore(":memory:")
    yield s
    s.close()


def _slept_values(inject_anomaly: bool, n: int = 50) -> list[float]:
    random.seed(1234)
    tracer = Tracer()
    slept: list[float] = []
    with patch("oracle.simulator.time.sleep", side_effect=slept.append):
        for _ in range(n):
            _simulate_llm_call(tracer, None, inject_anomaly=inject_anomaly)
    return slept


def test_normal_llm_latency_stays_in_base_range():
    slept = _slept_values(inject_anomaly=False)
    assert min(slept) >= 0.05 * LATENCY_SCALE
    assert max(slept) <= 0.25 * LATENCY_SCALE


def test_injected_anomaly_makes_llm_call_much_slower():
    normal = _slept_values(inject_anomaly=False)
    injected = _slept_values(inject_anomaly=True)
    assert max(injected) > max(normal)
    assert sum(injected) / len(injected) > 3 * (sum(normal) / len(normal))


def test_llm_call_span_records_cost_and_model():
    tracer = Tracer()
    with patch("oracle.simulator.time.sleep"):
        _simulate_llm_call(tracer, None)
    (span,) = tracer.spans()
    assert span.kind == "llm_call"
    assert span.attributes["cost_usd"] > 0
    assert span.attributes["model"]


def test_generate_traces_creates_one_agent_run_per_run(store):
    random.seed(7)
    with patch("oracle.simulator.time.sleep"):
        generate_traces(store, num_runs=12, anomaly_rate=0.0, error_rate=0.0)
    assert len(store.get_spans(kind="agent_run")) == 12


def test_generate_traces_nests_llm_calls_under_their_run(store):
    random.seed(7)
    with patch("oracle.simulator.time.sleep"):
        generate_traces(store, num_runs=5, anomaly_rate=0.0, error_rate=0.0)
    run_ids = {s["span_id"] for s in store.get_spans(kind="agent_run")}
    llm_calls = store.get_spans(kind="llm_call")
    assert len(llm_calls) == 10  # two per run
    assert all(s["parent_id"] in run_ids for s in llm_calls)


def test_generate_traces_records_tool_failures_as_error_spans(store):
    random.seed(3)
    with patch("oracle.simulator.time.sleep"):
        generate_traces(store, num_runs=60, anomaly_rate=0.0, error_rate=1.0)
    errors = [s for s in store.get_spans(kind="tool_call") if s["status"] == "error"]
    assert errors
    assert all(s["error"] == "simulated tool failure" for s in errors)


def _simulate(drift_rate: float, seed: int, num_runs: int = 100) -> TraceStore:
    random.seed(seed)
    store = TraceStore(":memory:")
    with patch("oracle.simulator.time.sleep"):
        generate_traces(
            store, num_runs=num_runs, anomaly_rate=0.0, error_rate=0.0, drift_rate=drift_rate
        )
    return store


def test_llm_call_span_records_a_summary_output():
    random.seed(0)
    tracer = Tracer()
    with patch("oracle.simulator.time.sleep"):
        _simulate_llm_call(tracer, None)
    (span,) = tracer.spans()
    assert span.attributes["output"].startswith("summary:")
    assert "revenue" in span.attributes["output"]


def test_normal_outputs_stay_close_to_each_other():
    random.seed(0)
    tracer = Tracer()
    with patch("oracle.simulator.time.sleep"):
        for _ in range(30):
            _simulate_llm_call(tracer, None)
    outputs = [s.attributes["output"] for s in tracer.spans()]
    sims = [jaccard_similarity(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:]]
    assert sum(sims) / len(sims) > 0.5


def test_drifted_outputs_are_far_from_normal_ones():
    normal = NORMAL_OUTPUT.format(pct=5)
    assert all(jaccard_similarity(normal, d) < 0.1 for d in DRIFTED_OUTPUTS)


def test_inject_drift_uses_a_drifted_output():
    random.seed(0)
    tracer = Tracer()
    with patch("oracle.simulator.time.sleep"):
        _simulate_llm_call(tracer, None, inject_drift=True)
    assert tracer.spans()[0].attributes["output"] in DRIFTED_OUTPUTS


def test_drift_rate_one_drifts_every_first_llm_call():
    store = _simulate(drift_rate=1.0, seed=1, num_runs=10)
    outputs = [s["attributes"]["output"] for s in store.get_spans(kind="llm_call")]
    assert len(outputs) == 20
    assert sum(o in DRIFTED_OUTPUTS for o in outputs) == 10  # one per run
    store.close()


def test_no_drift_alerts_when_drift_rate_is_zero():
    for seed in range(5):
        store = _simulate(drift_rate=0.0, seed=seed)
        assert DriftDetector().scan(store, kind="llm_call") == []
        store.close()


def test_injected_drift_is_detected_without_false_alarms():
    caught = missed = false_alarms = 0
    for seed in range(5):
        store = _simulate(drift_rate=0.05, seed=seed)
        llm = store.get_spans(kind="llm_call", limit=5000)
        drifted = {s["span_id"] for s in llm if s["attributes"]["output"] in DRIFTED_OUTPUTS}
        flagged = {a.span_id for a in DriftDetector().scan(store, kind="llm_call")}
        caught += len(flagged & drifted)
        missed += len(drifted - flagged)
        false_alarms += len(flagged - drifted)
        store.close()

    assert caught > 0
    assert false_alarms == 0
    assert caught > 4 * missed  # a few at the start lack history to compare against
