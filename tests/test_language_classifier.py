from language_classifier import classify_transcript


def test_neutral_acknowledgement_preserves_language_without_loading_model():
    result = classify_transcript("hi")
    assert result["label"] == "neutral"
    assert result["reason"] == "neutral_acknowledgement"


def test_result_has_stable_schema_for_regular_transcript():
    result = classify_transcript("hello how are you")
    assert {"label", "confidence", "probabilities", "latency_ms", "available", "reason"} <= set(result)
    assert 0.0 <= result["confidence"] <= 1.0
