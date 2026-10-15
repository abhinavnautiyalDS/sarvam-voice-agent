"""Optional wrapper for the trained Hindi/English/Hinglish transcript classifier.

The joblib model is fetched from the public classifier repository on first use (or
loaded from LANGUAGE_CLASSIFIER_MODEL_PATH). We intentionally keep this optional:
a missing/corrupt classifier should not bring down the voice pipeline.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import httpx

MODEL_URL = os.getenv(
    "LANGUAGE_CLASSIFIER_MODEL_URL",
    "https://raw.githubusercontent.com/abhinavnautiyalDS/hindi-english-language-classifier/main/models/english_hinglish_classifier.joblib",
)
MODEL_PATH = Path(
    os.getenv("LANGUAGE_CLASSIFIER_MODEL_PATH", "models/english_hinglish_classifier.joblib")
)
_MODEL: Any | None = None
_MODEL_ERROR = ""
_LOCK = threading.Lock()


def load_model() -> Any | None:
    """Load model from disk; download to disk once if not present."""
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            if not MODEL_PATH.exists():
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                response = httpx.get(MODEL_URL, timeout=90, follow_redirects=True)
                response.raise_for_status()
                if len(response.content) < 1024:
                    raise ValueError("Downloaded classifier model looks unexpectedly small")
                temporary = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
                temporary.write_bytes(response.content)
                temporary.replace(MODEL_PATH)
            import joblib
            _MODEL = joblib.load(MODEL_PATH)
            _MODEL_ERROR = ""
        except Exception as exc:  # safe fallback; caller can inspect error
            _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
            _MODEL = None
    return _MODEL


def classify_transcript(text: str) -> dict[str, Any]:
    """Return label/confidence/probabilities with measured inference latency."""
    import time

    started = time.perf_counter()
    clean = " ".join(str(text or "").strip().lower().split())
    neutral_words = {"hello", "hi", "hey", "bye", "goodbye", "thanks", "thank you", "okay", "ok", "yes", "no", "sure", "please", "right", "correct", "great", "fine", "hmm", "hmm okay", "okay sir", "ok sir", "yes sir", "no sir", "thank you sir"}
    if clean in neutral_words:
        return {"label": "neutral", "confidence": 1.0, "probabilities": {"neutral": 1.0}, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "available": True, "reason": "neutral_acknowledgement"}
    if len(clean) < int(os.getenv("LANGUAGE_CLASSIFIER_MIN_CHARS", "4")):
        return {"label": "uncertain", "confidence": 0.0, "probabilities": {}, "latency_ms": 0.0, "available": True, "reason": "transcript_too_short"}
    model = load_model()

    if model is None:
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "probabilities": {},
            "latency_ms": 0.0,
            "available": False,
            "reason": _MODEL_ERROR or "model_not_loaded",
        }
    started = time.perf_counter()
    try:
        probabilities = model.predict_proba([clean])[0]
        classes = list(model.classes_)
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        threshold = float(os.getenv("LANGUAGE_CLASSIFIER_MIN_CONFIDENCE", "0.60"))
        label = str(classes[index]).lower()
        if label not in {"english", "hinglish", "other", "uncertain", "neutral"}:
            label = "uncertain"
        if confidence < threshold and label not in {"other", "neutral", "uncertain"}:
            label = "uncertain"
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                str(name): float(score)
                for name, score in zip(classes, probabilities)
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "available": True,
            "reason": "below_confidence_threshold" if label == "uncertain" else "",
        }
    except Exception as exc:
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "probabilities": {},
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
