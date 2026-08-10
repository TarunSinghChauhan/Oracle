"""
ORACLE — Observability & Anomaly Detection for LLM Agents

The intel layer that watches your agent fleet: traces every run,
flags anomalies in cost/latency/output drift, and pushes alerts
before things break in production.
"""

__version__ = "0.1.0"

from oracle.tracer import Tracer, Span
from oracle.storage import TraceStore

__all__ = ["Tracer", "Span", "TraceStore"]
