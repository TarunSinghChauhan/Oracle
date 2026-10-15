import time
import pytest

from oracle.tracer import Tracer, Span


def test_span_records_timing():
    tracer = Tracer()
    with tracer.span("op") as s:
        time.sleep(0.01)
    assert s.duration_ms is not None
    assert s.duration_ms >= 10

def test_span_default_status_ok():
    tracer = Tracer()
    with tracer.span("op") as s:
        pass
    assert s.status == "ok"
    assert s.error is None

def test_span_captures_error():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.span("op") as s:
            raise ValueError("boom")
    assert s.status == "error"
    assert s.error == "boom"

def test_nested_spans_share_trace_id():
    tracer = Tracer()
    with tracer.span("parent") as p:
        with tracer.span("child", parent=p) as c:
            pass
    assert c.trace_id == p.trace_id
    assert c.parent_id == p.span_id

def test_tracer_collects_all_spans():
    tracer = Tracer()
    with tracer.span("a"):
        pass
    with tracer.span("b"):
        pass
    assert len(tracer.spans()) == 2

def test_span_attributes_stored():
    tracer = Tracer()
    with tracer.span("op", attributes={"tokens": 42}) as s:
        pass
    assert s.attributes["tokens"] == 42

def test_span_to_dict_includes_duration():
    tracer = Tracer()
    with tracer.span("op") as s:
        pass
    d = s.to_dict()
    assert "duration_ms" in d
    assert d["name"] == "op"


def test_unfinished_span_has_no_duration():
    span = Span(span_id="s1", name="op", start_time=100.0)
    assert span.end_time is None
    assert span.duration_ms is None
    assert span.to_dict()["duration_ms"] is None


def test_finished_span_duration_is_in_milliseconds():
    span = Span(span_id="s1", name="op", start_time=100.0, end_time=100.25)
    assert span.duration_ms == pytest.approx(250.0)
