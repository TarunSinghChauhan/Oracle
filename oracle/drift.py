"""
drift.py — Output drift detection for LLM spans.

Compares an LLM call's output text against a rolling baseline of prior
outputs for the same span `name`, using lexical similarity (Jaccard on
word shingles). No embeddings API, no external calls — pure stdlib.

This catches the common real-world failure mode: a prompt/model that
used to produce short, on-topic answers suddenly starts producing
wildly different-shaped output (much longer, much shorter, totally
different vocabulary) — often a sign of a broken prompt, a model
swap, or upstream data drift.
"""

from __future__ import annotations

import functools
import statistics
from dataclasses import dataclass
from typing import Optional

from oracle.anomaly import Anomaly


@dataclass
class DriftResult:
    span_id: str
    name: str
    severity: str          # "warning" | "critical"
    similarity: float      # 0.0 (totally different) - 1.0 (identical)
    baseline_similarity: float
    message: str


@functools.lru_cache(maxsize=1024)
def _shingles(text: str, n: int = 3) -> set[str]:
    """Word n-grams — robust to word order/length changes better than raw word sets.

    Cached: the same recent outputs are compared again and again as the window
    slides. Callers must treat the returned set as read-only.
    """
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


@functools.lru_cache(maxsize=4096)
def _similarity(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def jaccard_similarity(a: str, b: str) -> float:
    """0.0 = no shared shingles, 1.0 = identical shingle sets."""
    # Symmetric, so order the pair to share one cache entry per combination.
    return _similarity(a, b) if a <= b else _similarity(b, a)


class DriftDetector:
    """
    Tracks a rolling window of recent outputs per span `name` and flags
    when a new output's similarity to its own recent history drops
    sharply below the pairwise-similarity baseline of that history.
    """

    def __init__(
        self,
        window: int = 10,
        min_samples: int = 3,
        drift_threshold: float = 0.5,
    ):
        self.window = window
        self.min_samples = min_samples
        self.drift_threshold = drift_threshold
        self._history: dict[str, list[str]] = {}

    def _baseline_similarity(self, name: str) -> Optional[float]:
        history = self._history.get(name, [])
        if len(history) < 2:
            return None
        pairs = []
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                pairs.append(jaccard_similarity(history[i], history[j]))
        return statistics.mean(pairs) if pairs else None

    def check(self, span_id: str, name: str, output_text: str) -> Optional[DriftResult]:
        """
        Check `output_text` against history for `name`, then record it
        into history regardless of outcome (so the window keeps moving).
        """
        history = self._history.setdefault(name, [])
        baseline = self._baseline_similarity(name)
        result = None

        if baseline is not None and len(history) >= self.min_samples:
            similarities = [jaccard_similarity(output_text, h) for h in history]
            avg_similarity = statistics.mean(similarities)

            if avg_similarity < baseline * self.drift_threshold:
                severity = "critical" if avg_similarity < baseline * 0.25 else "warning"
                result = DriftResult(
                    span_id=span_id,
                    name=name,
                    severity=severity,
                    similarity=round(avg_similarity, 3),
                    baseline_similarity=round(baseline, 3),
                    message=(
                        f"'{name}' output diverged from recent baseline "
                        f"(similarity {avg_similarity:.2f} vs baseline {baseline:.2f})"
                    ),
                )

        history.append(output_text)
        if len(history) > self.window:
            history.pop(0)

        return result

    def scan(self, store, kind: Optional[str] = None, limit: int = 2000) -> list[Anomaly]:
        """
        Check the stored spans for output drift and return it as anomalies.

        Reads each span's ``attributes["output"]`` (spans without a text output
        are skipped) oldest to newest, so every output is judged against the
        ones before it. Uses a fresh history with this detector's settings, so
        calling it neither depends on nor changes the detector's own state.
        Newest first, like ``AnomalyDetector.scan``.
        """
        detector = DriftDetector(self.window, self.min_samples, self.drift_threshold)
        found: list[Anomaly] = []
        for span in reversed(store.get_spans(kind=kind, limit=limit)):
            output = (span.get("attributes") or {}).get("output")
            if not isinstance(output, str):
                continue
            result = detector.check(span["span_id"], span["name"], output)
            if result:
                found.append(Anomaly(
                    span_id=result.span_id,
                    kind="drift",
                    severity=result.severity,
                    message=result.message,
                    value=result.similarity,
                    baseline=result.baseline_similarity,
                ))
        found.reverse()
        return found
