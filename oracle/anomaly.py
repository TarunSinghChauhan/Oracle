"""
anomaly.py — Rules-based anomaly detection over spans.

No LLM calls, no API keys. Pure statistics: rolling baselines +
threshold rules for latency, cost, and error-rate spikes.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional


@dataclass
class Anomaly:
    span_id: str
    kind: str            # "latency_spike" | "cost_spike" | "error_burst" | "drift"
    severity: str         # "warning" | "critical"
    message: str
    value: float
    baseline: float


class _RunningStats:
    """
    Incremental mean / population stdev over a growing sample.

    Sums are kept as exact fractions, so identical values give a variance of
    exactly zero (the zero-variance case in ``_zscore`` depends on that) and
    results do not drift the way float running sums do.
    """

    __slots__ = ("n", "_s1", "_s2")

    def __init__(self) -> None:
        self.n = 0
        self._s1 = Fraction(0)
        self._s2 = Fraction(0)

    def add(self, x: float) -> None:
        x = Fraction(x)
        self.n += 1
        self._s1 += x
        self._s2 += x * x

    @property
    def mean(self) -> float:
        return float(self._s1 / self.n)

    @property
    def stdev(self) -> float:
        mean = self._s1 / self.n
        return math.sqrt(self._s2 / self.n - mean * mean)


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
        return self._zscore_from(value, statistics.mean(samples), statistics.pstdev(samples))

    @staticmethod
    def _zscore_from(value: float, mean: float, stdev: float) -> float:
        if stdev == 0:
            # Zero-variance baseline: any value above the mean is infinitely
            # anomalous. Both callers flag upward spikes only, so a value at
            # or below the mean (a faster or cheaper span) is not anomalous.
            return float("inf") if value > mean else 0.0
        return (value - mean) / stdev

    def check_latency(self, span: dict, history: list[dict]) -> Optional[Anomaly]:
        durations = [
            h["duration_ms"] for h in history
            if h["name"] == span["name"] and h["duration_ms"] is not None
        ]
        if span["duration_ms"] is None:
            return None
        z = self._zscore(span["duration_ms"], durations)
        if z is None or z < self.latency_z_threshold:
            return None
        return self._latency_anomaly(span, z, statistics.mean(durations))

    def _latency_anomaly(self, span: dict, z: float, baseline: float) -> Anomaly:
        return Anomaly(
            span_id=span["span_id"],
            kind="latency_spike",
            severity="critical" if z > self.latency_z_threshold * 1.5 else "warning",
            message=f"'{span['name']}' took {span['duration_ms']:.0f}ms "
                    f"({z:.1f} std devs above baseline)",
            value=span["duration_ms"],
            baseline=baseline,
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
        return self._cost_anomaly(span, cost, z, statistics.mean(costs))

    def _cost_anomaly(self, span: dict, cost: float, z: float, baseline: float) -> Anomaly:
        return Anomaly(
            span_id=span["span_id"],
            kind="cost_spike",
            severity="critical" if z > self.cost_z_threshold * 1.5 else "warning",
            message=f"'{span['name']}' cost ${cost:.4f} "
                    f"({z:.1f} std devs above baseline)",
            value=cost,
            baseline=baseline,
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

        # Walk oldest -> newest, judging each span against running stats of the
        # older spans with the same name, then folding it into those stats.
        # This is linear in the number of spans; re-reading the whole history
        # for every span made a scan quadratic (seconds for 2000 spans, and the
        # dashboard rescans every few seconds).
        latency_stats: dict[str, _RunningStats] = {}
        cost_stats: dict[str, _RunningStats] = {}
        per_span: dict[int, list[Anomaly]] = {}
        for i in range(len(spans) - 1, -1, -1):
            span = spans[i]
            lat = latency_stats.setdefault(span["name"], _RunningStats())
            cst = cost_stats.setdefault(span["name"], _RunningStats())
            found: list[Anomaly] = []

            duration = span["duration_ms"]
            if duration is not None:
                if lat.n >= self.min_samples:
                    z = self._zscore_from(duration, lat.mean, lat.stdev)
                    if z >= self.latency_z_threshold:
                        found.append(self._latency_anomaly(span, z, lat.mean))
                lat.add(duration)

            cost = span.get("attributes", {}).get("cost_usd")
            if cost is not None:
                if cst.n >= self.min_samples:
                    z = self._zscore_from(cost, cst.mean, cst.stdev)
                    if z >= self.cost_z_threshold:
                        found.append(self._cost_anomaly(span, cost, z, cst.mean))
                cst.add(cost)

            if found:
                per_span[i] = found

        # Report newest first, as before.
        anomalies: list[Anomaly] = [a for i in sorted(per_span) for a in per_span[i]]
        burst = self.check_error_burst(spans)
        if burst:
            anomalies.append(burst)
        return anomalies
