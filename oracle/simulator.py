"""
simulator.py — Generates synthetic agent traces so ORACLE is fully
demoable without any API keys or real LLM calls.

Simulates realistic latency/cost distributions with occasional
injected anomalies, so the anomaly detector has something to catch.
"""

from __future__ import annotations

import random
import time

from oracle.tracer import Tracer
from oracle.storage import TraceStore

TOOL_NAMES = ["web_search", "code_exec", "file_read", "vector_lookup"]
LLM_MODELS = ["claude-sonnet", "claude-haiku", "gpt-4o-mini"]

# Simulated LLM latency is `base_latency * LATENCY_SCALE` seconds. Scaling (rather
# than capping) keeps injected anomalies proportionally slower than normal calls
# while a demo run still finishes quickly.
LATENCY_SCALE = 0.1


def _simulate_llm_call(tracer: Tracer, parent, inject_anomaly: bool = False):
    base_latency = random.uniform(0.05, 0.25)
    base_cost = random.uniform(0.001, 0.01)
    if inject_anomaly:
        base_latency *= random.uniform(5, 12)
        base_cost *= random.uniform(4, 10)

    with tracer.span(
        "llm_call",
        kind="llm_call",
        parent=parent,
        attributes={
            "model": random.choice(LLM_MODELS),
            "tokens": random.randint(100, 2000),
            "cost_usd": round(base_cost, 5),
        },
    ):
        time.sleep(base_latency * LATENCY_SCALE)


def _simulate_tool_call(tracer: Tracer, parent, force_error: bool = False):
    with tracer.span(
        "tool_call",
        kind="tool_call",
        parent=parent,
        attributes={"tool": random.choice(TOOL_NAMES)},
    ):
        time.sleep(random.uniform(0.001, 0.01))
        if force_error:
            raise RuntimeError("simulated tool failure")


def generate_traces(
    store: TraceStore,
    num_runs: int = 50,
    anomaly_rate: float = 0.08,
    error_rate: float = 0.05,
) -> None:
    """Populate `store` with `num_runs` synthetic agent runs."""
    for i in range(num_runs):
        tracer = Tracer(store=store)
        inject_anomaly = random.random() < anomaly_rate
        force_error = random.random() < error_rate

        try:
            with tracer.span("agent_run", kind="agent_run", attributes={"run_index": i}) as run:
                _simulate_llm_call(tracer, run, inject_anomaly=inject_anomaly)
                for _ in range(random.randint(1, 3)):
                    try:
                        _simulate_tool_call(tracer, run, force_error=force_error and random.random() < 0.3)
                    except RuntimeError:
                        pass  # error already recorded on the span
                _simulate_llm_call(tracer, run)
        except RuntimeError:
            pass


if __name__ == "__main__":
    store = TraceStore("oracle_traces.db")
    generate_traces(store, num_runs=100)
    print("Generated 100 synthetic agent runs into oracle_traces.db")
    store.close()
