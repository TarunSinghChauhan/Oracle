import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from oracle.anomaly import AnomalyDetector
from oracle.cli import main
from oracle.storage import TraceStore
from oracle.tracer import Span


def _save(store, i, duration_ms=100.0, kind="llm_call", status="ok"):
    start = 1_000.0 + i
    store.save_span(Span(
        span_id=f"s{i}", name=kind, start_time=start,
        end_time=start + duration_ms / 1000.0, trace_id="t",
        kind=kind, status=status, attributes={},
    ))


@pytest.fixture
def healthy_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = TraceStore(path)
    for i in range(10):
        _save(store, i, duration_ms=100.0 + i)
    store.close()
    yield path
    os.remove(path)


@pytest.fixture
def spiky_db(healthy_db):
    store = TraceStore(healthy_db)
    _save(store, 10, duration_ms=5000.0)
    store.close()
    return healthy_db


def test_scan_healthy_db_reports_no_anomalies(healthy_db, capsys):
    assert main(["scan", healthy_db]) == 0
    assert "No anomalies found." in capsys.readouterr().out


def test_scan_prints_anomalies_and_exits_zero_by_default(spiky_db, capsys):
    assert main(["scan", spiky_db]) == 0
    assert "latency_spike" in capsys.readouterr().out


def test_fail_on_warning_exits_one_when_anomaly_found(spiky_db):
    assert main(["scan", spiky_db, "--fail-on", "warning"]) == 1


def test_fail_on_critical_ignores_lower_severities(healthy_db):
    store = TraceStore(healthy_db)
    # About 3.7 std devs above the baseline: a warning, not a critical spike.
    _save(store, 10, duration_ms=115.0)
    found = AnomalyDetector().scan(store)
    store.close()
    assert [a.severity for a in found] == ["warning"]
    assert main(["scan", healthy_db, "--fail-on", "critical"]) == 0
    assert main(["scan", healthy_db, "--fail-on", "warning"]) == 1


def test_fail_on_is_zero_for_healthy_db(healthy_db):
    assert main(["scan", healthy_db, "--fail-on", "warning"]) == 0


def test_kind_filter_only_scans_that_kind(spiky_db, capsys):
    assert main(["scan", spiky_db, "--kind", "tool_call"]) == 0
    assert "No anomalies found." in capsys.readouterr().out


def test_missing_database_is_an_error_and_is_not_created(tmp_path, capsys):
    missing = tmp_path / "typo.db"
    assert main(["scan", str(missing)]) == 2
    assert "no such trace database" in capsys.readouterr().err
    assert not missing.exists()


def test_webhook_flag_posts_instead_of_printing(spiky_db, capsys):
    with patch("oracle.alerts.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value.status = 200
        assert main(["scan", spiky_db, "--webhook", "https://hooks.example.com/x"]) == 0

    assert urlopen.call_count == 1
    out = capsys.readouterr().out
    assert "Sent 1 of 1 alerts" in out
    assert "latency_spike" not in out


def test_webhook_reports_failed_deliveries(spiky_db, capsys):
    with patch("oracle.alerts.urllib.request.urlopen", side_effect=OSError("down")):
        assert main(["scan", spiky_db, "--webhook", "https://hooks.example.com/x"]) == 0
    assert "Sent 0 of 1 alerts" in capsys.readouterr().out


def test_command_is_required():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_python_dash_m_oracle_scan_runs(spiky_db):
    result = subprocess.run(
        [sys.executable, "-m", "oracle", "scan", spiky_db, "--fail-on", "warning"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "latency_spike" in result.stdout
