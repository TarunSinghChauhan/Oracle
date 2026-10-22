"""
cli.py — Command line entry point: ``python -m oracle scan``.

Scans a trace database for anomalies and prints them (or posts them to a
webhook). Exits non-zero when anomalies at or above ``--fail-on`` severity
are found, so it can gate a CI job or cron check. With ``--watch`` it keeps
scanning on an interval and reports each anomaly once, when it first appears.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from oracle.alerts import AlertManager, ConsoleSink, WebhookSink
from oracle.anomaly import Anomaly
from oracle.scan import scan_all
from oracle.storage import TraceStore

SEVERITY_RANK = {"warning": 1, "critical": 2}


def _positive_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


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
    scan.add_argument(
        "--watch", type=_positive_seconds, metavar="SECONDS",
        help="keep scanning every SECONDS and report each anomaly once, when it "
             "first appears (Ctrl+C to stop)",
    )
    return parser


def _db_exists(path: str) -> bool:
    # TraceStore would silently create an empty database for a mistyped path,
    # so check first and report it instead of scanning nothing.
    if Path(path).is_file():
        return True
    print(f"oracle: no such trace database: {path}", file=sys.stderr)
    return False


def _find_anomalies(args: argparse.Namespace) -> list[Anomaly]:
    store = TraceStore(args.db)
    try:
        return scan_all(store, kind=args.kind)
    finally:
        store.close()


def _alert_manager(args: argparse.Namespace) -> AlertManager:
    return AlertManager(sink=WebhookSink(args.webhook) if args.webhook else ConsoleSink())


def _scan(args: argparse.Namespace) -> int:
    if not _db_exists(args.db):
        return 2

    anomalies = _find_anomalies(args)
    sent = _alert_manager(args).dispatch(anomalies)

    if not anomalies:
        print("No anomalies found.")
    elif args.webhook:
        print(f"Sent {sent} of {len(anomalies)} alerts to the webhook.")

    if args.fail_on:
        threshold = SEVERITY_RANK[args.fail_on]
        if any(SEVERITY_RANK.get(a.severity, 0) >= threshold for a in anomalies):
            return 1
    return 0


def _anomaly_key(anomaly: Anomaly) -> tuple:
    # An error burst is anchored to whichever span is newest, so its span_id
    # changes with every new span while the burst goes on; key it by kind alone.
    if anomaly.kind == "error_burst":
        return (anomaly.kind,)
    return (anomaly.kind, anomaly.span_id)


def _watch(
    args: argparse.Namespace,
    sleep: Optional[Callable[[float], None]] = None,
    max_scans: Optional[int] = None,
) -> int:
    sleep = sleep or time.sleep  # looked up per call so tests can patch time.sleep
    manager = _alert_manager(args)
    seen: set[tuple] = set()
    scans = 0
    print(f"Watching {args.db} every {args.watch:g}s (Ctrl+C to stop)", flush=True)
    try:
        while True:
            if not _db_exists(args.db):
                return 2
            current = {_anomaly_key(a): a for a in _find_anomalies(args)}
            new = [a for key, a in current.items() if key not in seen]
            # Remember only what is still reported, so an anomaly that clears
            # and later comes back is alerted again.
            seen = set(current)
            if new:
                sent = manager.dispatch(new)
                if args.webhook:
                    print(f"Sent {sent} of {len(new)} new alerts to the webhook.", flush=True)
            scans += 1
            if max_scans is not None and scans >= max_scans:
                return 0
            sleep(args.watch)
    except KeyboardInterrupt:
        print("Stopped.")
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "scan":
        if args.watch is not None:
            if args.fail_on:
                parser.error("--fail-on cannot be used with --watch")
            return _watch(args)
        return _scan(args)
    return 2  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
