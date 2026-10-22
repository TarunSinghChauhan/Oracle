import random
import time

from oracle.drift import DriftDetector, jaccard_similarity


def test_identical_text_similarity_is_one():
    assert jaccard_similarity("hello world foo", "hello world foo") == 1.0

def test_completely_different_text_similarity_is_low():
    sim = jaccard_similarity("the cat sat on the mat", "quantum physics is hard today")
    assert sim < 0.2

def test_empty_strings_are_identical():
    assert jaccard_similarity("", "") == 1.0

def test_no_drift_with_insufficient_history():
    det = DriftDetector(min_samples=3)
    result = det.check("s1", "llm_call", "the weather today is sunny")
    assert result is None

def test_no_drift_flagged_for_consistent_outputs():
    det = DriftDetector(min_samples=3, window=10)
    consistent_outputs = [
        "the weather today is sunny and warm",
        "the weather today is cloudy and cool",
        "the weather today is rainy and mild",
        "the weather today is sunny and hot",
    ]
    results = [det.check(f"s{i}", "llm_call", text) for i, text in enumerate(consistent_outputs)]
    assert all(r is None for r in results)

def test_drift_detected_on_sudden_topic_change():
    det = DriftDetector(min_samples=3, window=10, drift_threshold=0.5)
    for i, text in enumerate([
        "the weather today is sunny and warm outside",
        "the weather today is cloudy and cool outside",
        "the weather today is rainy and mild outside",
    ]):
        det.check(f"s{i}", "llm_call", text)

    result = det.check("s_drift", "llm_call", "quantum entanglement violates local realism theories")
    assert result is not None
    assert result.severity in ("warning", "critical")

def test_history_window_respects_max_size():
    det = DriftDetector(min_samples=2, window=3)
    for i in range(10):
        det.check(f"s{i}", "llm_call", f"output number {i} here")
    assert len(det._history["llm_call"]) == 3

def test_separate_span_names_have_independent_history():
    det = DriftDetector(min_samples=3, window=10)
    for i in range(3):
        det.check(f"a{i}", "summarize", "the summary is short and factual")
    result = det.check("b1", "translate", "totally unrelated content here now")
    # translate has no history yet, so no drift should fire
    assert result is None


def test_text_shorter_than_one_shingle_compares_as_a_whole_phrase():
    assert jaccard_similarity("ok done", "ok done") == 1.0
    assert jaccard_similarity("ok done", "all good") == 0.0


def test_short_text_never_matches_longer_text_by_accident():
    assert jaccard_similarity("ok", "the quick brown fox jumps") == 0.0


def test_empty_text_versus_non_empty_text_is_zero():
    assert jaccard_similarity("", "some real output here") == 0.0
    assert jaccard_similarity("some real output here", "") == 0.0


def test_whitespace_only_text_counts_as_empty():
    assert jaccard_similarity("   \n\t ", "") == 1.0


def test_similarity_ignores_case_and_extra_whitespace():
    assert jaccard_similarity("The  Quick brown FOX", "the quick   brown fox") == 1.0


def test_detector_flags_output_that_goes_empty():
    det = DriftDetector(window=10, min_samples=3, drift_threshold=0.5)
    for i in range(4):
        det.check(f"s{i}", "summarize", "the quarterly report shows revenue growth")
    result = det.check("s4", "summarize", "")
    assert result is not None
    assert result.severity == "critical"
    assert result.similarity == 0.0


class ListStore:
    """Minimal stand-in for TraceStore: scan() only calls get_spans()."""

    def __init__(self, spans):
        self.spans = spans  # newest first, like TraceStore.get_spans

    def get_spans(self, kind=None, limit=1000):
        spans = [s for s in self.spans if kind is None or s["kind"] == kind]
        return spans[:limit]


def _span(i, output, name="summarize", kind="llm_call"):
    attrs = {} if output is None else {"output": output}
    return {"span_id": f"s{i}", "name": name, "kind": kind, "attributes": attrs}


def _store(outputs, **kw):
    """Spans numbered oldest (0) to newest, stored newest first."""
    return ListStore([_span(i, o, **kw) for i, o in reversed(list(enumerate(outputs)))])


STEADY = "the quarterly report shows steady revenue growth across all regions this year"
BROKEN = "error upstream service unavailable please retry the request later today"


def test_scan_flags_a_sudden_change_in_output():
    found = DriftDetector().scan(_store([STEADY] * 5 + [BROKEN]))

    assert [(a.kind, a.span_id, a.severity) for a in found] == [("drift", "s5", "critical")]
    assert found[0].value == 0.0
    assert found[0].baseline == 1.0
    assert "diverged from recent baseline" in found[0].message


def test_scan_is_quiet_for_consistent_outputs():
    assert DriftDetector().scan(_store([STEADY] * 8)) == []


def test_scan_skips_spans_without_text_output():
    outputs = [STEADY, None, STEADY, STEADY, None, STEADY, BROKEN]
    found = DriftDetector().scan(_store(outputs))
    assert [a.span_id for a in found] == ["s6"]
    # No output at all means nothing to compare.
    assert DriftDetector().scan(_store([None] * 10)) == []


def test_scan_handles_an_empty_store():
    assert DriftDetector().scan(_store([])) == []


def test_scan_respects_the_kind_filter():
    store = _store([STEADY] * 5 + [BROKEN], kind="tool_call")
    assert DriftDetector().scan(store, kind="llm_call") == []
    assert len(DriftDetector().scan(store, kind="tool_call")) == 1


def test_scan_keeps_a_separate_history_per_span_name():
    spans = []
    for i in range(6):  # interleave two names; only "b" changes at the end
        spans.append(_span(2 * i, STEADY, name="a"))
        spans.append(_span(2 * i + 1, STEADY if i < 5 else BROKEN, name="b"))
    found = DriftDetector().scan(ListStore(list(reversed(spans))))
    assert [(a.span_id) for a in found] == ["s11"]


def test_scan_reports_newest_first():
    found = DriftDetector(window=4).scan(_store([STEADY] * 4 + [BROKEN] + [STEADY] * 4 + [BROKEN]))
    assert [a.span_id for a in found] == sorted((a.span_id for a in found), key=lambda s: -int(s[1:]))
    assert len(found) >= 2


def test_scan_does_not_use_or_change_the_detectors_own_history():
    detector = DriftDetector()
    detector.check("live", "summarize", STEADY)
    before = {k: list(v) for k, v in detector._history.items()}

    first = detector.scan(_store([STEADY] * 5 + [BROKEN]))
    second = detector.scan(_store([STEADY] * 5 + [BROKEN]))

    assert first == second and len(first) == 1
    assert detector._history == before


def test_scan_matches_feeding_check_one_output_at_a_time():
    rng = random.Random(5)
    calm = "revenue growth steady across regions while costs remained flat this quarter".split()
    odd = "unexpected failure retry later upstream timeout gateway refused connection".split()
    outputs = [" ".join(rng.choices(calm if rng.random() < 0.8 else odd, k=14)) for _ in range(150)]
    names = [rng.choice(["a", "b", "c"]) for _ in outputs]
    spans = [_span(i, o, name=n) for i, (o, n) in enumerate(zip(outputs, names))]

    reference = DriftDetector()
    expected = []
    for span in spans:
        result = reference.check(span["span_id"], span["name"], span["attributes"]["output"])
        if result:
            expected.append((span["span_id"], result.severity, result.message))

    found = DriftDetector().scan(ListStore(list(reversed(spans))))
    assert expected, "fixture should produce some drift"
    assert [(a.span_id, a.severity, a.message) for a in reversed(found)] == expected


def test_similarity_is_symmetric():
    a, b = "one two three four five", "three four five six seven"
    assert jaccard_similarity(a, b) == jaccard_similarity(b, a)


def test_scan_cost_stays_low_for_many_outputs():
    # Recomputing every pairwise similarity from scratch took about 3.4s for
    # 2000 outputs; with cached shingles and pairs it takes a fraction of a second.
    rng = random.Random(1)
    vocab = "the quarterly report shows revenue growth across regions while costs flat".split()
    outputs = [" ".join(rng.choices(vocab, k=60)) for _ in range(2000)]

    started = time.perf_counter()
    DriftDetector().scan(_store(outputs))
    assert time.perf_counter() - started < 2.0
