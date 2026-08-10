"""
storage.py — SQLite persistence for spans.

Zero external dependencies, zero API keys. Just a local .db file
the dashboard and anomaly detector read from.
"""

from __future__ import annotations

import json
import sqlite3
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
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def save_span(self, span: Span) -> None:
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

        cur = self._conn.execute(query, params)
        cols = [c[0] for c in cur.description]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            d["attributes"] = json.loads(d["attributes"]) if d["attributes"] else {}
            rows.append(d)
        return rows

    def close(self) -> None:
        self._conn.close()
