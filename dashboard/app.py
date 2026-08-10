"""
dashboard/app.py — The Batcomputer console.

A small Flask app that reads directly from a TraceStore (SQLite) and
surfaces: recent spans, live anomaly scan results, and basic run stats.
No API keys, no external calls — pure read layer over your own trace data.

Run:
    python dashboard/app.py --db oracle_traces.db
Then open http://127.0.0.1:5050
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.storage import TraceStore
from oracle.anomaly import AnomalyDetector

app = Flask(__name__)
app.config["DB_PATH"] = "oracle_traces.db"


def get_store() -> TraceStore:
    return TraceStore(app.config["DB_PATH"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    store = get_store()
    spans = store.get_spans(limit=5000)
    store.close()

    if not spans:
        return jsonify({
            "total_spans": 0, "total_runs": 0, "error_count": 0,
            "avg_latency_ms": 0, "total_cost_usd": 0,
        })

    runs = [s for s in spans if s["kind"] == "agent_run"]
    errors = [s for s in spans if s["status"] == "error"]
    durations = [s["duration_ms"] for s in spans if s["duration_ms"] is not None]
    costs = [
        s["attributes"].get("cost_usd", 0) for s in spans
        if s["attributes"].get("cost_usd") is not None
    ]

    return jsonify({
        "total_spans": len(spans),
        "total_runs": len(runs),
        "error_count": len(errors),
        "avg_latency_ms": round(statistics.mean(durations), 2) if durations else 0,
        "total_cost_usd": round(sum(costs), 4),
    })


@app.route("/api/spans")
def api_spans():
    store = get_store()
    spans = store.get_spans(limit=100)
    store.close()
    return jsonify(spans)


@app.route("/api/anomalies")
def api_anomalies():
    store = get_store()
    detector = AnomalyDetector()
    anomalies = detector.scan(store)
    store.close()
    return jsonify([
        {
            "span_id": a.span_id,
            "kind": a.kind,
            "severity": a.severity,
            "message": a.message,
            "value": a.value,
            "baseline": a.baseline,
        }
        for a in anomalies
    ])


def main():
    parser = argparse.ArgumentParser(description="ORACLE dashboard")
    parser.add_argument("--db", default="oracle_traces.db", help="Path to trace store DB")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    app.config["DB_PATH"] = args.db
    print(f"ORACLE dashboard running at http://127.0.0.1:{args.port}")
    print(f"Reading traces from: {args.db}")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
