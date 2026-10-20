import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from oracle.tracer import Tracer
from oracle.storage import TraceStore
from dashboard.app import app


@pytest.fixture
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    store = TraceStore(path)
    tracer = Tracer(store=store)
    with tracer.span("agent_run", kind="agent_run") as run:
        with tracer.span("llm_call", kind="llm_call", parent=run,
                          attributes={"cost_usd": 0.004}):
            pass
    store.close()

    app.config["DB_PATH"] = path
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

    os.remove(path)


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200

def test_stats_endpoint_returns_expected_keys(client):
    resp = client.get("/api/stats")
    data = resp.get_json()
    assert resp.status_code == 200
    for key in ("total_spans", "total_runs", "error_count", "avg_latency_ms", "total_cost_usd"):
        assert key in data

def test_stats_counts_are_correct(client):
    data = client.get("/api/stats").get_json()
    assert data["total_spans"] == 2
    assert data["total_runs"] == 1

def test_spans_endpoint_returns_list(client):
    data = client.get("/api/spans").get_json()
    assert isinstance(data, list)
    assert len(data) == 2

def test_anomalies_endpoint_returns_list(client):
    data = client.get("/api/anomalies").get_json()
    assert isinstance(data, list)

def test_stats_on_empty_store(tmp_path):
    empty_db = str(tmp_path / "empty.db")
    TraceStore(empty_db).close()
    app.config["DB_PATH"] = empty_db
    with app.test_client() as c:
        data = c.get("/api/stats").get_json()
        assert data["total_spans"] == 0
        assert data["total_cost_usd"] == 0


def test_stats_count_every_span_not_just_the_newest_page(tmp_path):
    # /api/stats used to be computed from the newest 5000 spans only, so a busy
    # store reported 5000 spans and understated runs, errors and cost.
    path = str(tmp_path / "busy.db")
    store = TraceStore(path)
    rows = [
        (f"s{i}", "t", None, "op", "agent_run" if i % 10 == 0 else "llm_call",
         "error" if i % 7 == 0 else "ok", 1_000.0 + i, 1_000.1 + i, 100.0,
         json.dumps({"cost_usd": 0.01}), None)
        for i in range(6000)
    ]
    store._conn.executemany("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    store._conn.commit()
    store.close()

    app.config["DB_PATH"] = path
    with app.test_client() as c:
        data = c.get("/api/stats").get_json()

    assert data == {
        "total_spans": 6000,
        "total_runs": 600,
        "error_count": 858,
        "avg_latency_ms": 100.0,
        "total_cost_usd": 60.0,
    }
