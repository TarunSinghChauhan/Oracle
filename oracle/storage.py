"""
storage.py — SQLite persistence for spans.

Zero external dependencies, zero API keys. Just a local .db file
the dashboard and anomaly detector read from.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from oracle.tracer import Span

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    parent_id   TEXT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL,
    duration_ms REAL,
    attributes  TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind);
CREATE INDEX IF NOT EXISTS idx_spans_start ON spans(start_time);
"""


class TraceStore:
    def __init__(self, db_path: str = "oracle_traces.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True) if Path(db_path).parent != Path("") else None
        # Spans are written from whichever thread a callback fires on (e.g.
        # LangChain runs sync handlers in a thread pool), so the connection
        # must be shareable; the lock serialises access to it.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def save_span(self, span: Span) -> None:
        with self._lock:
            self._save_span(span)

    def _save_span(self, span: Span) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_id, name, kind, status,
                start_time, end_time, duration_ms, attributes, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                span.span_id,
                span.trace_id,
                span.parent_id,
                span.name,
                span.kind,
                span.status,
                span.start_time,
                span.end_time,
                span.duration_ms,
                json.dumps(span.attributes),
                span.error,
            ),
        )
        self._conn.commit()

    def get_spans(
        self,
        kind: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        query = "SELECT * FROM spans WHERE 1=1"
        params: list = []
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        if trace_id:
            query += " AND trace_id = ?"
            params.append(trace_id)
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(query, params)
            cols = [c[0] for c in cur.description]
            fetched = cur.fetchall()
        rows = []
        for row in fetched:
            d = dict(zip(cols, row))
            d["attributes"] = json.loads(d["attributes"]) if d["attributes"] else {}
            rows.append(d)
        return rows

    def stats(self) -> dict:
        """Totals over every stored span, not just the newest page of them."""
        with self._lock:
            total, runs, errors, avg_latency = self._conn.execute(
                "SELECT COUNT(*),"
                " COALESCE(SUM(kind = 'agent_run'), 0),"
                " COALESCE(SUM(status = 'error'), 0),"
                " AVG(duration_ms)"
                " FROM spans"
            ).fetchone()
            attribute_rows = self._conn.execute(
                "SELECT attributes FROM spans WHERE attributes LIKE '%cost_usd%'"
            ).fetchall()

        cost = 0.0
        for (raw,) in attribute_rows:
            value = json.loads(raw).get("cost_usd")
            if value is not None:
                cost += value
        return {
            "total_spans": total,
            "total_runs": runs,
            "error_count": errors,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else 0,
            "total_cost_usd": round(cost, 4),
        }

    def prune(self, before: float, dry_run: bool = False) -> int:
        """
        Delete spans that started before `before` (a Unix timestamp) and return
        how many. With ``dry_run`` nothing is deleted, only counted.

        Spans are pruned one by one by start time, not by trace: a Tracer keeps
        one trace id for its whole lifetime, so a long-running agent would
        otherwise never become old enough to prune.
        """
        with self._lock:
            if dry_run:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM spans WHERE start_time < ?", (before,)
                ).fetchone()
                return row[0]
            cur = self._conn.execute("DELETE FROM spans WHERE start_time < ?", (before,))
            self._conn.commit()
            return cur.rowcount

    def vacuum(self) -> None:
        """Give the space freed by prune() back to the file system."""
        with self._lock:
            self._conn.commit()
            self._conn.execute("VACUUM")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
