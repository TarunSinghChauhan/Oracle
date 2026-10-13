import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from oracle.alerts import AlertManager, ConsoleSink, WebhookSink
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


def _fake_response(status: int):
    resp = MagicMock()
    resp.status = status
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_webhook_sink_posts_slack_compatible_payload():
    sink = WebhookSink("https://hooks.example.com/abc", timeout=2.5)
    with patch("oracle.alerts.urllib.request.urlopen", return_value=_fake_response(200)) as urlopen:
        assert sink.send(_mk_anomaly()) is True

    req = urlopen.call_args.args[0]
    assert req.full_url == "https://hooks.example.com/abc"
    assert req.get_method() == "POST"
    assert req.get_header("Content-type") == "application/json"
    text = json.loads(req.data)["text"]
    assert "WARNING" in text
    assert "latency_spike" in text
    assert "test anomaly" in text
    assert urlopen.call_args.kwargs["timeout"] == 2.5


@pytest.mark.parametrize(
    "status, expected",
    [(200, True), (204, True), (299, True), (300, False), (500, False)],
)
def test_webhook_sink_treats_only_2xx_as_success(status, expected):
    sink = WebhookSink("https://hooks.example.com/abc")
    with patch("oracle.alerts.urllib.request.urlopen", return_value=_fake_response(status)):
        assert sink.send(_mk_anomaly()) is expected


def test_webhook_sink_returns_false_on_network_error():
    sink = WebhookSink("https://hooks.example.com/abc")
    with patch(
        "oracle.alerts.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        assert sink.send(_mk_anomaly()) is False


def test_webhook_sink_default_timeout_is_five_seconds():
    assert WebhookSink("https://hooks.example.com/abc").timeout == 5.0


def test_alert_manager_counts_only_successful_sends():
    class FlakySink:
        def __init__(self):
            self.results = iter([True, False, True])

        def send(self, anomaly):
            return next(self.results)

    manager = AlertManager(sink=FlakySink())
    assert manager.dispatch([_mk_anomaly(), _mk_anomaly(), _mk_anomaly()]) == 2
