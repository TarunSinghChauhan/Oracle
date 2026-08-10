"""
examples/instrument_agent.py — Minimal real-world usage example.

Shows how to wire ORACLE's Tracer into an actual agent loop:
LLM call -> tool call -> LLM call, with cost/latency tracked per span.
Run this, then launch the dashboard to see it show up live.

    python examples/instrument_agent.py
    python dashboard/app.py
"""

import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.tracer import Tracer
from oracle.storage import TraceStore


def fake_llm_call(prompt: str) -> str:
    """Stand-in for a real LLM call — swap with your actual client."""
    time.sleep(random.uniform(0.01, 0.05))
    return f"response to: {prompt[:30]}"


def fake_tool_call(query: str) -> str:
    """Stand-in for a real tool/function call."""
    time.sleep(random.uniform(0.005, 0.02))
    return f"result for: {query}"


def run_agent(tracer: Tracer, user_input: str):
    with tracer.span("agent_run", kind="agent_run", attributes={"input": user_input}) as run:
        with tracer.span(
            "llm_call", kind="llm_call", parent=run,
            attributes={"model": "claude-sonnet", "cost_usd": round(random.uniform(0.001, 0.008), 5)},
        ):
            plan = fake_llm_call(user_input)

        with tracer.span("tool_call", kind="tool_call", parent=run, attributes={"tool": "web_search"}):
            result = fake_tool_call(plan)

        with tracer.span(
            "llm_call", kind="llm_call", parent=run,
            attributes={"model": "claude-sonnet", "cost_usd": round(random.uniform(0.001, 0.008), 5)},
        ):
            final = fake_llm_call(result)

    return final


if __name__ == "__main__":
    store = TraceStore("oracle_traces.db")
    tracer = Tracer(store=store)

    questions = [
        "What's the weather in Jaipur?",
        "Summarize this quarter's revenue",
        "Find flights to Tokyo",
    ]
    for q in questions:
        answer = run_agent(tracer, q)
        print(f"Q: {q}\nA: {answer}\n")

    store.close()
    print("Traces written to oracle_traces.db — run the dashboard to view them.")
