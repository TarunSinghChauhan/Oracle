import os
import tempfile

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from oracle.tracer import Tracer
from oracle.storage import TraceStore
from oracle.integrations.langchain import OracleCallbackHandler


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = TraceStore(path)
    yield s
    s.close()
    os.remove(path)


def test_fake_llm_call_creates_llm_span(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)
    llm = FakeListChatModel(responses=["hello there"])

    llm.invoke("hi", config={"callbacks": [handler]})

    spans = store.get_spans(kind="llm_call")
    assert len(spans) == 1
    assert spans[0]["status"] == "ok"

def test_llm_span_has_duration_recorded(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)
    llm = FakeListChatModel(responses=["response text"])

    llm.invoke("test prompt", config={"callbacks": [handler]})

    spans = store.get_spans(kind="llm_call")
    assert spans[0]["duration_ms"] is not None
    assert spans[0]["duration_ms"] >= 0

def test_tool_call_creates_tool_span(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    add_numbers.invoke({"a": 2, "b": 3}, config={"callbacks": [handler]})

    spans = store.get_spans(kind="tool_call")
    assert len(spans) == 1
    assert spans[0]["attributes"]["tool"] == "add_numbers"

def test_multiple_llm_calls_create_separate_spans(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)
    llm = FakeListChatModel(responses=["first", "second"])

    llm.invoke("prompt one", config={"callbacks": [handler]})
    llm.invoke("prompt two", config={"callbacks": [handler]})

    spans = store.get_spans(kind="llm_call")
    assert len(spans) == 2

def test_handler_cleans_up_open_spans_after_call(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)
    llm = FakeListChatModel(responses=["done"])

    llm.invoke("hi", config={"callbacks": [handler]})

    assert len(handler._open_spans) == 0
    assert len(handler._spans) == 0

def test_import_error_message_is_clear():
    """The module should be importable when langchain-core is present."""
    from oracle.integrations.langchain import OracleCallbackHandler
    assert OracleCallbackHandler is not None


def test_chain_span_uses_run_name(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)
    chain = RunnableLambda(lambda x: x).with_config(run_name="prepare_input")

    chain.invoke("hi", config={"callbacks": [handler]})

    names = [s["name"] for s in store.get_spans(kind="agent_run")]
    assert names == ["prepare_input"]


def test_chain_spans_keep_distinct_names_and_nesting(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)
    chain = (
        RunnableLambda(lambda x: x).with_config(run_name="step_one")
        | RunnableLambda(lambda x: x).with_config(run_name="step_two")
    )

    chain.invoke("hi", config={"callbacks": [handler]})

    spans = store.get_spans(kind="agent_run")
    by_name = {s["name"]: s for s in spans}
    assert {"step_one", "step_two"} <= set(by_name)
    assert "chain_run" not in by_name
    outer = by_name["RunnableSequence"]
    assert by_name["step_one"]["parent_id"] == outer["span_id"]
    assert by_name["step_two"]["parent_id"] == outer["span_id"]


def test_chain_start_falls_back_to_serialized_name_then_default():
    handler = OracleCallbackHandler(Tracer())
    handler.on_chain_start({"name": "legacy_chain"}, {}, run_id="r1")
    handler.on_chain_start(None, {}, run_id="r2")

    assert handler._spans["r1"].name == "legacy_chain"
    assert handler._spans["r2"].name == "chain_run"


def test_tool_error_marks_span_as_error(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)

    @tool
    def explode(x: int) -> int:
        """Always fails."""
        raise ValueError("boom")

    with pytest.raises(ValueError):
        explode.invoke({"x": 1}, config={"callbacks": [handler]})

    spans = store.get_spans(kind="tool_call")
    assert len(spans) == 1
    assert spans[0]["status"] == "error"
    assert spans[0]["error"] == "boom"
    assert handler._open_spans == {} and handler._spans == {}


def test_chain_error_marks_span_as_error(store):
    tracer = Tracer(store=store)
    handler = OracleCallbackHandler(tracer)

    def fail(_):
        raise RuntimeError("chain broke")

    chain = RunnableLambda(fail).with_config(run_name="flaky_step")

    with pytest.raises(RuntimeError):
        chain.invoke("hi", config={"callbacks": [handler]})

    spans = store.get_spans(kind="agent_run")
    assert [(s["name"], s["status"], s["error"]) for s in spans] == [
        ("flaky_step", "error", "chain broke")
    ]


def test_llm_error_marks_span_as_error(store):
    handler = OracleCallbackHandler(Tracer(store=store))
    handler.on_llm_start({"name": "fake"}, ["p"], run_id="r1")
    handler.on_llm_error(TimeoutError("rate limited"), run_id="r1")

    spans = store.get_spans(kind="llm_call")
    assert len(spans) == 1
    assert spans[0]["status"] == "error"
    assert spans[0]["error"] == "rate limited"
    assert handler._spans == {}


def test_llm_start_records_model_and_prompt_count(store):
    handler = OracleCallbackHandler(Tracer(store=store))
    handler.on_llm_start({"name": "gpt-x"}, ["a", "b"], run_id="r1")
    handler.on_llm_end(object(), run_id="r1")

    attrs = store.get_spans(kind="llm_call")[0]["attributes"]
    assert attrs == {"model": "gpt-x", "prompt_count": 2}


def test_chat_model_start_defaults_model_name(store):
    handler = OracleCallbackHandler(Tracer(store=store))
    handler.on_chat_model_start(None, [[]], run_id="r1")
    handler.on_llm_end(object(), run_id="r1")

    assert store.get_spans(kind="llm_call")[0]["attributes"] == {"model": "unknown_model"}


def test_llm_end_records_total_tokens(store):
    from types import SimpleNamespace

    handler = OracleCallbackHandler(Tracer(store=store))
    handler.on_llm_start({"name": "m"}, ["p"], run_id="r1")
    response = SimpleNamespace(llm_output={"token_usage": {"total_tokens": 42}})
    handler.on_llm_end(response, run_id="r1")

    assert store.get_spans(kind="llm_call")[0]["attributes"]["tokens"] == 42


def test_tool_start_defaults_tool_name(store):
    handler = OracleCallbackHandler(Tracer(store=store))
    handler.on_tool_start(None, "input", run_id="r1")
    handler.on_tool_end("out", run_id="r1")

    assert store.get_spans(kind="tool_call")[0]["attributes"] == {"tool": "unknown_tool"}


def test_ending_an_unknown_run_is_a_noop(store):
    handler = OracleCallbackHandler(Tracer(store=store))
    handler.on_tool_end("out", run_id="never-started")
    handler.on_chain_error(ValueError("x"), run_id="never-started")

    assert store.get_spans() == []


def test_nested_llm_span_is_parented_to_chain_span(store):
    handler = OracleCallbackHandler(Tracer(store=store))
    handler.on_chain_start(None, {}, run_id="chain", name="pipeline")
    handler.on_llm_start({"name": "m"}, ["p"], run_id="llm", parent_run_id="chain")
    handler.on_llm_end(object(), run_id="llm", parent_run_id="chain")
    handler.on_chain_end({}, run_id="chain")

    chain = store.get_spans(kind="agent_run")[0]
    llm = store.get_spans(kind="llm_call")[0]
    assert llm["parent_id"] == chain["span_id"]


def test_output_is_not_captured_by_default(store):
    handler = OracleCallbackHandler(Tracer(store=store))
    FakeListChatModel(responses=["secret answer"]).invoke("hi", config={"callbacks": [handler]})

    assert "output" not in store.get_spans(kind="llm_call")[0]["attributes"]


def test_capture_output_stores_the_response_text(store):
    handler = OracleCallbackHandler(Tracer(store=store), capture_output=True)
    FakeListChatModel(responses=["hello there"]).invoke("hi", config={"callbacks": [handler]})

    assert store.get_spans(kind="llm_call")[0]["attributes"]["output"] == "hello there"


def test_captured_output_is_truncated(store):
    handler = OracleCallbackHandler(Tracer(store=store), capture_output=True, max_output_chars=5)
    FakeListChatModel(responses=["hello there"]).invoke("hi", config={"callbacks": [handler]})

    assert store.get_spans(kind="llm_call")[0]["attributes"]["output"] == "hello"


def test_capture_output_keeps_an_empty_response(store):
    # An output that goes empty is itself drift worth seeing.
    handler = OracleCallbackHandler(Tracer(store=store), capture_output=True)
    FakeListChatModel(responses=[""]).invoke("hi", config={"callbacks": [handler]})

    assert store.get_spans(kind="llm_call")[0]["attributes"]["output"] == ""


def test_capture_output_ignores_responses_without_generations(store):
    handler = OracleCallbackHandler(Tracer(store=store), capture_output=True)
    for bad in (object(), type("R", (), {"generations": []})(), type("R", (), {"generations": [[]]})()):
        handler.on_llm_start({"name": "m"}, ["p"], run_id=f"r{id(bad)}")
        handler.on_llm_end(bad, run_id=f"r{id(bad)}")

    spans = store.get_spans(kind="llm_call")
    assert len(spans) == 3
    assert all("output" not in s["attributes"] for s in spans)


def test_capture_output_still_records_token_usage(store):
    from types import SimpleNamespace

    handler = OracleCallbackHandler(Tracer(store=store), capture_output=True)
    handler.on_llm_start({"name": "m"}, ["p"], run_id="r1")
    response = SimpleNamespace(
        llm_output={"token_usage": {"total_tokens": 7}},
        generations=[[SimpleNamespace(text="ok")]],
    )
    handler.on_llm_end(response, run_id="r1")

    attrs = store.get_spans(kind="llm_call")[0]["attributes"]
    assert attrs["tokens"] == 7 and attrs["output"] == "ok"


def test_max_output_chars_must_be_positive():
    with pytest.raises(ValueError):
        OracleCallbackHandler(Tracer(), capture_output=True, max_output_chars=0)


def test_captured_outputs_feed_drift_detection_end_to_end(store):
    from oracle.scan import scan_all

    steady = "the quarterly report shows steady revenue growth across all regions this year"
    broken = "error upstream service unavailable please retry the request later today"
    handler = OracleCallbackHandler(Tracer(store=store), capture_output=True)
    llm = FakeListChatModel(responses=[steady] * 5 + [broken])

    for _ in range(6):
        llm.invoke("summarize", config={"callbacks": [handler]})

    drift = [a for a in scan_all(store) if a.kind == "drift"]
    assert len(drift) == 1
    assert drift[0].severity == "critical"
