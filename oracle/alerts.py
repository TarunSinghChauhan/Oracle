"""
alerts.py — Push anomalies to a webhook (Slack-compatible payload shape).

No API key required to build/test: only needed if you point
webhook_url at a real Slack/Discord endpoint. Defaults to a
no-op "console" sink so it's fully testable offline.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional, Protocol

from oracle.anomaly import Anomaly


class AlertSink(Protocol):
    def send(self, anomaly: Anomaly) -> bool: ...


class ConsoleSink:
    """Default sink — prints alerts. No network calls, no keys."""

    def __init__(self):
        self.sent: list[Anomaly] = []

    def send(self, anomaly: Anomaly) -> bool:
        self.sent.append(anomaly)
        print(f"[{anomaly.severity.upper()}] {anomaly.kind}: {anomaly.message}")
        return True


class WebhookSink:
    """Posts a Slack-compatible JSON payload to any incoming webhook URL."""

    def __init__(self, webhook_url: str, timeout: float = 5.0):
        self.webhook_url = webhook_url
        self.timeout = timeout

    def send(self, anomaly: Anomaly) -> bool:
        payload = {
            "text": f":rotating_light: *{anomaly.severity.upper()}* — {anomaly.kind}\n{anomaly.message}"
        }
        req = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False


class AlertManager:
    def __init__(self, sink: Optional[AlertSink] = None):
        self.sink = sink or ConsoleSink()

    def dispatch(self, anomalies: list[Anomaly]) -> int:
        sent = 0
        for a in anomalies:
            if self.sink.send(a):
                sent += 1
        return sent
