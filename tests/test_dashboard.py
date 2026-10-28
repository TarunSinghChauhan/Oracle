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


def test_anomalies_endpoint_includes_output_drift(tmp_path):
    path = str(tmp_path / "drift.db")
    store = TraceStore(path)
    tracer = Tracer(store=store)
    steady = "the quarterly report shows steady revenue growth across all regions this year"
    broken = "error upstream service unavailable please retry the request later today"
    for i in range(6):
        with tracer.span("summarize", kind="llm_call",
                         attributes={"output": broken if i == 5 else steady}):
            pass
    store.close()

    app.config["DB_PATH"] = path
    with app.test_client() as c:
        data = c.get("/api/anomalies").get_json()

    drift = [a for a in data if a["kind"] == "drift"]
    assert len(drift) == 1
    assert drift[0]["severity"] == "critical"


# Interpolations in the page script that are numbers or fixed strings computed
# from the stats payload, so they cannot carry markup. Anything else that is
# put into innerHTML must go through esc().
SAFE_UNESCAPED = {
    "s.total_spans",
    "s.total_runs",
    "s.error_count",
    "s.error_count > 0 ? 'warn' : ''",
    "s.avg_latency_ms",
    "s.total_cost_usd",
    "s.duration_ms ? s.duration_ms.toFixed(1) + 'ms' : '\u2014'",
}


def _template_script():
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "index.html").read_text()
    return html.split("<script>", 1)[1].split("</script>", 1)[0]


def test_template_escapes_every_text_field():
    # Span names, kinds, statuses and anomaly messages come from traced data and
    # were inserted into the page with innerHTML unescaped (stored XSS).
    import re

    script = _template_script()
    assert "function esc(" in script
    expressions = re.findall(r"\$\{([^}]*)\}", script)
    unescaped = {e.strip() for e in expressions if not e.strip().startswith("esc(")}
    assert unescaped <= SAFE_UNESCAPED, f"unescaped interpolations: {unescaped - SAFE_UNESCAPED}"

    escaped = {e.strip() for e in expressions if e.strip().startswith("esc(")}
    assert escaped == {
        "esc(s.name)", "esc(s.kind)", "esc(s.status)",
        "esc(a.kind)", "esc(a.severity)", "esc(a.message)",
    }


def test_main_refuses_a_missing_database(tmp_path, capsys):
    from unittest.mock import patch

    from dashboard.app import main

    missing = tmp_path / "typo.db"
    with patch.object(app, "run") as run:
        assert main(["--db", str(missing)]) == 2

    assert "no such trace database" in capsys.readouterr().err
    assert not missing.exists()  # no stray empty database
    run.assert_not_called()


def test_main_serves_an_existing_database(tmp_path, capsys):
    from unittest.mock import patch

    from dashboard.app import main

    path = str(tmp_path / "ok.db")
    TraceStore(path).close()
    with patch.object(app, "run") as run:
        assert main(["--db", path, "--port", "6123"]) == 0

    run.assert_called_once_with(port=6123, debug=False)
    assert app.config["DB_PATH"] == path
    assert "http://127.0.0.1:6123" in capsys.readouterr().out


def test_main_defaults_to_port_5050(tmp_path):
    from unittest.mock import patch

    from dashboard.app import main

    path = str(tmp_path / "ok.db")
    TraceStore(path).close()
    with patch.object(app, "run") as run:
        assert main(["--db", path]) == 0

    run.assert_called_once_with(port=5050, debug=False)


@pytest.mark.parametrize("route, failing", [
    ("/api/stats", ("store", "stats")),
    ("/api/spans", ("store", "get_spans")),
    ("/api/anomalies", ("scan_all", None)),
])
def test_failing_requests_still_close_the_store(monkeypatch, route, failing):
    from unittest.mock import MagicMock, patch

    monkeypatch.setitem(app.config, "TESTING", True)
    store = MagicMock()
    target, method = failing
    if target == "store":
        getattr(store, method).side_effect = RuntimeError("boom")
    with patch("dashboard.app.get_store", return_value=store):
        if target == "scan_all":
            with patch("dashboard.app.scan_all", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError):
                    app.test_client().get(route)
        else:
            with pytest.raises(RuntimeError):
                app.test_client().get(route)

    store.close.assert_called_once()
