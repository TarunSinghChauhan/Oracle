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
