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
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.storage import TraceStore
from oracle.scan import scan_all

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
    try:
        return jsonify(store.stats())
    finally:
        store.close()


@app.route("/api/spans")
def api_spans():
    store = get_store()
    try:
        spans = store.get_spans(limit=100)
    finally:
        store.close()
    return jsonify(spans)


@app.route("/api/anomalies")
def api_anomalies():
    store = get_store()
    try:
        anomalies = scan_all(store)
    finally:
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ORACLE dashboard")
    parser.add_argument("--db", default="oracle_traces.db", help="Path to trace store DB")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args(argv)

    # TraceStore creates a missing database on first use, so a mistyped path
    # would start a dashboard that shows "no spans yet" and leave an empty file
    # behind. Fail up front instead.
    if not Path(args.db).is_file():
        print(f"oracle: no such trace database: {args.db}", file=sys.stderr)
        return 2

    app.config["DB_PATH"] = args.db
    print(f"ORACLE dashboard running at http://127.0.0.1:{args.port}")
    print(f"Reading traces from: {args.db}")
    app.run(port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
