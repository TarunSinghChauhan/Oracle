"""
anomaly.py — Rules-based anomaly detection over spans.

No LLM calls, no API keys. Pure statistics: rolling baselines +
threshold rules for latency, cost, and error-rate spikes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class Anomaly:
    span_id: str
    kind: str            # "latency_spike" | "cost_spike" | "error_burst" | "drift"
    severity: str         # "warning" | "critical"
    message: str
    value: float
    baseline: float


class AnomalyDetector:
    """
    Compares incoming spans against a rolling baseline (mean + stdev)
    built from prior spans of the same `kind`/`name`.
    """

    def __init__(
        self,
        latency_z_threshold: float = 3.0,
        cost_z_threshold: float = 3.0,
        min_samples: int = 5,
    ):
        self.latency_z_threshold = latency_z_threshold
        self.cost_z_threshold = cost_z_threshold
        self.min_samples = min_samples

    def _zscore(self, value: float, samples: list[float]) -> Optional[float]:
        if len(samples) < self.min_samples:
            return None
        mean = statistics.mean(samples)
        stdev = statistics.pstdev(samples)
        if stdev == 0:
            # Zero-variance baseline: any deviation is infinitely
            # anomalous, no deviation is not anomalous at all.
            return 0.0 if value == mean else float("inf")
        return (value - mean) / stdev

    def check_latency(self, span: dict, history: list[dict]) -> Optional[Anomaly]:
        durations = [
            h["duration_ms"] for h in history
            if h["name"] == span["name"] and h["duration_ms"] is not None
        ]
        z = self._zscore(span["duration_ms"], durations)
        if z is None or z < self.latency_z_threshold:
            return None
        return Anomaly(
            span_id=span["span_id"],
            kind="latency_spike",
            severity="critical" if z > self.latency_z_threshold * 1.5 else "warning",
            message=f"'{span['name']}' took {span['duration_ms']:.0f}ms "
                    f"({z:.1f} std devs above baseline)",
            value=span["duration_ms"],
            baseline=statistics.mean(durations),
        )

    def check_cost(self, span: dict, history: list[dict]) -> Optional[Anomaly]:
        cost = span.get("attributes", {}).get("cost_usd")
        if cost is None:
            return None
        costs = [
            h["attributes"].get("cost_usd") for h in history
            if h["name"] == span["name"] and h["attributes"].get("cost_usd") is not None
        ]
        z = self._zscore(cost, costs)
        if z is None or z < self.cost_z_threshold:
            return None
        return Anomaly(
            span_id=span["span_id"],
            kind="cost_spike",
            severity="critical" if z > self.cost_z_threshold * 1.5 else "warning",
            message=f"'{span['name']}' cost ${cost:.4f} "
                    f"({z:.1f} std devs above baseline)",
            value=cost,
            baseline=statistics.mean(costs),
        )

    def check_error_burst(self, recent: list[dict], window: int = 20, rate_threshold: float = 0.3) -> Optional[Anomaly]:
        sample = recent[:window]
        if len(sample) < self.min_samples:
            return None
        error_rate = sum(1 for s in sample if s["status"] == "error") / len(sample)
        if error_rate < rate_threshold:
            return None
        return Anomaly(
            span_id=sample[0]["span_id"],
            kind="error_burst",
            severity="critical" if error_rate > 0.5 else "warning",
            message=f"Error rate {error_rate:.0%} over last {len(sample)} spans",
            value=error_rate,
            baseline=rate_threshold,
        )

    def scan(self, store, kind: Optional[str] = None) -> list[Anomaly]:
        """Run all checks against everything currently in the store."""
        spans = store.get_spans(kind=kind, limit=2000)
        anomalies: list[Anomaly] = []
        for i, span in enumerate(spans):
            history = spans[i + 1:]  # older spans (DESC order)
            for check in (self.check_latency, self.check_cost):
                result = check(span, history)
                if result:
                    anomalies.append(result)
        burst = self.check_error_burst(spans)
        if burst:
            anomalies.append(burst)
        return anomalies
