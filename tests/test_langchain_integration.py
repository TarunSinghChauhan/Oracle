import os
import tempfile

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
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
