from __future__ import annotations

import os
import time
import unicodedata
from pathlib import Path

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "abhinavnautiyalDS/script-switch-classifier"
BRANCH = "full-project-backup"
MODEL_DIR = Path(os.getenv("SCRIPT_CLASSIFIER_CACHE", ".script_classifier_cache"))
MODEL_PATH = "artifacts/bert_tiny_script_classifier/model"
MAX_LENGTH = 128

_FILES = (
    "config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
)
_model = None
_tokenizer = None
_device = None


def _ensure_model_files() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{MODEL_PATH}"
    for filename in _FILES:
        target = MODEL_DIR / filename
        if target.exists() and target.stat().st_size:
            continue
        response = requests.get(f"{base_url}/{filename}", timeout=180)
        if response.status_code == 404 and filename in {"special_tokens_map.json", "vocab.txt"}:
            continue
        response.raise_for_status()
        target.write_bytes(response.content)
    for required in ("config.json", "model.safetensors", "tokenizer.json"):
        target = MODEL_DIR / required
        if not target.exists() or target.stat().st_size == 0:
            raise FileNotFoundError(f"Failed to download required classifier file: {required}")
    return MODEL_DIR


def _load():
    global _model, _tokenizer, _device
    if _model is None:
        path = _ensure_model_files()
        _tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        _model.eval()
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
    return _tokenizer, _model, _device


def classify_transcript(text: str) -> dict:
    """Classify dominant writing script: DEVANAGARI or LATIN."""
    cleaned = " ".join(unicodedata.normalize("NFC", str(text or "")).split())
    if not cleaned:
        return {"label": "unavailable", "confidence": None, "probabilities": {},
                "latency_ms": 0.0, "available": False, "reason": "empty_text"}
    try:
        tokenizer, model, device = _load()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        encoded = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            probs_tensor = torch.softmax(model(**encoded).logits, dim=-1)[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        values = probs_tensor.detach().cpu().tolist()
        id2label = model.config.id2label
        scores = {str(id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))).upper(): float(p)
                  for i, p in enumerate(values)}
        if set(scores) != {"DEVANAGARI", "LATIN"}:
            raise ValueError(f"Unexpected classifier labels: {list(scores)}")
        label = max(scores, key=scores.get)
        return {"label": label, "confidence": scores[label], "probabilities": scores,
                "latency_ms": round(latency_ms, 3), "available": True, "reason": ""}
    except Exception as exc:
        return {"label": "unavailable", "confidence": None, "probabilities": {},
                "latency_ms": 0.0, "available": False,
                "reason": f"{type(exc).__name__}: {exc}"}
