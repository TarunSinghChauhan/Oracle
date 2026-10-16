"""
cli.py — Command line entry point: ``python -m oracle scan``.

Scans a trace database for anomalies and prints them (or posts them to a
webhook). Exits non-zero when anomalies at or above ``--fail-on`` severity
are found, so it can gate a CI job or cron check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from oracle.alerts import AlertManager, ConsoleSink, WebhookSink
from oracle.anomaly import AnomalyDetector
from oracle.storage import TraceStore

SEVERITY_RANK = {"warning": 1, "critical": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle", description="ORACLE - observability for LLM agents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a trace database for anomalies")
    scan.add_argument(
        "db", nargs="?", default="oracle_traces.db",
        help="path to the SQLite trace database (default: oracle_traces.db)",
    )
    scan.add_argument("--kind", help="only scan spans of this kind, e.g. llm_call")
    scan.add_argument(
        "--webhook", metavar="URL",
        help="post anomalies to this Slack-compatible webhook instead of printing",
    )
    scan.add_argument(
        "--fail-on", choices=sorted(SEVERITY_RANK),
        help="exit with status 1 if any anomaly is at least this severe",
    )
    return parser


def _scan(args: argparse.Namespace) -> int:
    # TraceStore would silently create an empty database for a mistyped path,
    # so check first and report it instead of scanning nothing.
    if not Path(args.db).is_file():
        print(f"oracle: no such trace database: {args.db}", file=sys.stderr)
        return 2

    store = TraceStore(args.db)
    try:
        anomalies = AnomalyDetector().scan(store, kind=args.kind)
    finally:
        store.close()

    sink = WebhookSink(args.webhook) if args.webhook else ConsoleSink()
    sent = AlertManager(sink=sink).dispatch(anomalies)

    if not anomalies:
        print("No anomalies found.")
    elif args.webhook:
        print(f"Sent {sent} of {len(anomalies)} alerts to the webhook.")

    if args.fail_on:
        threshold = SEVERITY_RANK[args.fail_on]
        if any(SEVERITY_RANK.get(a.severity, 0) >= threshold for a in anomalies):
            return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        return _scan(args)
    return 2  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
