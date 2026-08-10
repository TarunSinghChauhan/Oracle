from oracle.alerts import AlertManager, ConsoleSink
from oracle.anomaly import Anomaly


def _mk_anomaly():
    return Anomaly(
        span_id="s1",
        kind="latency_spike",
        severity="warning",
        message="test anomaly",
        value=500.0,
        baseline=100.0,
    )

def test_console_sink_records_sent():
    sink = ConsoleSink()
    result = sink.send(_mk_anomaly())
    assert result is True
    assert len(sink.sent) == 1

def test_alert_manager_dispatches_all():
    sink = ConsoleSink()
    manager = AlertManager(sink=sink)
    anomalies = [_mk_anomaly(), _mk_anomaly()]
    sent = manager.dispatch(anomalies)
    assert sent == 2
    assert len(sink.sent) == 2

def test_alert_manager_defaults_to_console_sink():
    manager = AlertManager()
    assert isinstance(manager.sink, ConsoleSink)

def test_dispatch_empty_list():
    manager = AlertManager(sink=ConsoleSink())
    assert manager.dispatch([]) == 0
