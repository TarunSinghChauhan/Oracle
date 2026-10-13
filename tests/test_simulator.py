import random
from unittest.mock import patch

import pytest

from oracle.simulator import LATENCY_SCALE, _simulate_llm_call, generate_traces
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
