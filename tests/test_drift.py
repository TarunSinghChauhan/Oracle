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
