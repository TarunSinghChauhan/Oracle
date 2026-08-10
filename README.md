# ORACLE

**Observability & anomaly detection for LLM agents.**

Barbara Gordon watched over Batman's whole operation from the clock tower —
feeding him real-time intel, flagging threats before they became disasters.
ORACLE does the same for your agent fleet: it traces every run, watches
cost/latency/error patterns against a rolling baseline, and raises the
alarm the moment something drifts.

No API keys required. No external LLM calls. Pure Python + SQLite.

## Why

Most "LLM observability" tooling assumes you're already paying for a
hosted platform. ORACLE is the opposite bet: a lightweight, dependency-free
tracer + anomaly detector you can drop into any agent pipeline (LangChain,
LangGraph, a raw `while` loop — doesn't matter) and run entirely offline.

## Architecture

```
your agent code
      │  tracer.span("agent_run") / tracer.span("llm_call")
      ▼
  oracle.Tracer  ──────────────►  oracle.TraceStore (SQLite)
                                          │
                                          ▼
                                oracle.AnomalyDetector
                                (rolling z-score baselines:
                                 latency, cost, error rate)
                                          │
                                          ▼
                                  oracle.AlertManager
                                (console sink, or webhook
                                 → Slack/Discord)
```

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic traces (no API keys needed — pure simulation)
python -m oracle.simulator

# Scan for anomalies
python -c "
from oracle.storage import TraceStore
from oracle.anomaly import AnomalyDetector

store = TraceStore('oracle_traces.db')
anomalies = AnomalyDetector().scan(store)
for a in anomalies:
    print(f'[{a.severity.upper()}] {a.kind}: {a.message}')
"
```

## Instrumenting a real agent

```python
from oracle.tracer import Tracer
from oracle.storage import TraceStore

store = TraceStore("oracle_traces.db")
tracer = Tracer(store=store)

with tracer.span("agent_run", kind="agent_run") as run:
    with tracer.span("llm_call", kind="llm_call", parent=run,
                      attributes={"model": "claude-sonnet", "tokens": 512, "cost_usd": 0.003}):
        response = call_your_llm(prompt)

    with tracer.span("tool_call", kind="tool_call", parent=run,
                      attributes={"tool": "web_search"}):
        result = run_your_tool(query)
```

Every span is written to SQLite immediately — no batching, no buffering,
so you can query `oracle_traces.db` mid-run.

See `examples/instrument_agent.py` for a complete runnable example
(LLM call → tool call → LLM call), then launch the dashboard to watch
it show up live.

## Dashboard

A local Flask console reads directly from `TraceStore` — no external
services, no keys:

```bash
python -m oracle.simulator      # or examples/instrument_agent.py for real-shaped data
python dashboard/app.py
```

Open `http://127.0.0.1:5050` to see live spans, run stats, and an
anomaly feed.

## Anomaly detection

Rolling z-score baselines per span `name`, no LLM judge required:

- **Latency spikes** — a span takes N std devs longer than its own history
- **Cost spikes** — a span's `cost_usd` attribute blows past baseline
- **Error bursts** — error rate over the last N spans crosses a threshold

Thresholds are configurable (`AnomalyDetector(latency_z_threshold=3.0, ...)`).

## Alerting

```python
from oracle.alerts import AlertManager, WebhookSink

# Console (default, no network calls — safe for CI/tests)
manager = AlertManager()

# Or point at a real Slack/Discord incoming webhook
manager = AlertManager(sink=WebhookSink("https://hooks.slack.com/services/..."))
manager.dispatch(anomalies)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

24 tests covering tracer span lifecycle, SQLite persistence, anomaly
z-score edge cases (including zero-variance baselines), and alert dispatch.

## Roadmap

- [ ] Live dashboard (the "Batcomputer console") reading from `TraceStore`
- [ ] Drift detection on LLM output text (embedding distance vs. baseline)
- [ ] Ingestion adapters for LangChain/LangGraph callback handlers
- [ ] Optional LLM-powered plain-English anomaly explanations (bolt-on, not core)

## Status

Early-stage — core tracer, storage, anomaly detection, and alerting are
functional and tested. Dashboard and drift detection are next.
