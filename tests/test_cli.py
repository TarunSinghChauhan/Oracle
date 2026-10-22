import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from oracle.anomaly import AnomalyDetector
from oracle.cli import _watch, build_parser, main
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


def _watch_args(db, **overrides):
    args = build_parser().parse_args(["scan", db, "--watch", "1"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_watch_alerts_once_for_a_persisting_anomaly(spiky_db, capsys):
    assert _watch(_watch_args(spiky_db), sleep=lambda _: None, max_scans=3) == 0
    out = capsys.readouterr().out
    assert out.count("latency_spike") == 1


def test_watch_alerts_again_when_a_new_anomaly_appears(spiky_db, capsys):
    def add_second_spike(_seconds):
        store = TraceStore(spiky_db)
        _save(store, 11, duration_ms=100_000.0)
        store.close()

    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 1:
            add_second_spike(seconds)

    assert _watch(_watch_args(spiky_db), sleep=sleep, max_scans=3) == 0
    out = capsys.readouterr().out
    assert out.count("latency_spike") == 2
    assert calls == [1.0, 1.0]


def test_watch_does_not_realert_an_ongoing_error_burst(tmp_path, capsys):
    path = str(tmp_path / "burst.db")
    store = TraceStore(path)
    for i in range(10):
        _save(store, i, status="error" if i >= 3 else "ok")
    store.close()

    def add_another_error(_seconds):
        s = TraceStore(path)
        _save(s, 100, status="error")  # a new newest span: the burst continues
        s.close()

    assert _watch(_watch_args(path), sleep=add_another_error, max_scans=3) == 0
    assert capsys.readouterr().out.count("error_burst") == 1


def test_watch_webhook_posts_only_new_anomalies(spiky_db, capsys):
    args = _watch_args(spiky_db, webhook="https://hooks.example.com/x")
    with patch("oracle.alerts.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value.status = 200
        assert _watch(args, sleep=lambda _: None, max_scans=3) == 0

    assert urlopen.call_count == 1
    assert capsys.readouterr().out.count("Sent 1 of 1 new alerts") == 1


def test_watch_stays_quiet_when_nothing_is_wrong(healthy_db, capsys):
    assert _watch(_watch_args(healthy_db), sleep=lambda _: None, max_scans=2) == 0
    out = capsys.readouterr().out
    assert "Watching" in out
    assert "No anomalies" not in out and "latency_spike" not in out


def test_watch_stops_cleanly_on_ctrl_c(healthy_db, capsys):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    assert _watch(_watch_args(healthy_db), sleep=interrupt) == 0
    assert "Stopped." in capsys.readouterr().out


def test_watch_reports_a_database_that_disappears(tmp_path, capsys):
    path = tmp_path / "gone.db"
    store = TraceStore(str(path))
    store.close()

    def delete_db(_seconds):
        path.unlink()

    assert _watch(_watch_args(str(path)), sleep=delete_db, max_scans=5) == 2
    assert "no such trace database" in capsys.readouterr().err
    assert not path.exists()


def test_watch_missing_database_is_an_error(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "nope.db"), "--watch", "1"]) == 2
    assert "no such trace database" in capsys.readouterr().err


def test_main_watch_runs_until_interrupted(healthy_db, capsys):
    with patch("oracle.cli.time.sleep", side_effect=KeyboardInterrupt):
        assert main(["scan", healthy_db, "--watch", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "every 0.5s" in out and "Stopped." in out


@pytest.mark.parametrize("seconds", ["0", "-3", "soon"])
def test_watch_interval_must_be_a_positive_number(healthy_db, seconds):
    with pytest.raises(SystemExit) as exc:
        main(["scan", healthy_db, "--watch", seconds])
    assert exc.value.code == 2


def test_watch_cannot_be_combined_with_fail_on(healthy_db):
    with pytest.raises(SystemExit) as exc:
        main(["scan", healthy_db, "--watch", "5", "--fail-on", "warning"])
    assert exc.value.code == 2


def test_scan_reports_output_drift(tmp_path, capsys):
    path = str(tmp_path / "drift.db")
    store = TraceStore(path)
    steady = "the quarterly report shows steady revenue growth across all regions this year"
    broken = "error upstream service unavailable please retry the request later today"
    for i in range(6):
        start = 1_000.0 + i
        store.save_span(Span(
            span_id=f"s{i}", name="summarize", start_time=start, end_time=start + 0.1,
            trace_id="t", kind="llm_call",
            attributes={"output": broken if i == 5 else steady},
        ))
    store.close()

    assert main(["scan", path, "--fail-on", "warning"]) == 1
    out = capsys.readouterr().out
    assert "drift" in out and "diverged from recent baseline" in out
