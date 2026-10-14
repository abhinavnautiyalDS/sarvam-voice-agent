from language_classifier import classify_transcript


def test_short_transcript_is_uncertain_without_loading_model():
    result = classify_transcript("hi")
    assert result["label"] == "uncertain"
    assert result["reason"] == "transcript_too_short"


def test_result_has_stable_schema_for_regular_transcript():
    result = classify_transcript("hello how are you")
    assert {"label", "confidence", "probabilities", "latency_ms", "available", "reason"} <= set(result)
    assert 0.0 <= result["confidence"] <= 1.0
