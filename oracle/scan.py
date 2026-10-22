"""
scan.py — Run every detector over a trace store.

One entry point for the CLI and the dashboard, so a new detector only has to
be added here to show up in both.
"""

from __future__ import annotations

from typing import Optional

from oracle.anomaly import Anomaly, AnomalyDetector
from oracle.drift import DriftDetector


def scan_all(store, kind: Optional[str] = None) -> list[Anomaly]:
    """Latency, cost and error-rate anomalies, then output drift."""
    return AnomalyDetector().scan(store, kind=kind) + DriftDetector().scan(store, kind=kind)
