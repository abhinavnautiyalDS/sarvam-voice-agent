from __future__ import annotations

import os
import time
import unicodedata
import zipfile
from pathlib import Path

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSIFIER_REPO = "abhinavnautiyalDS/multilingual-voice-language-classifier"
CLASSIFIER_BRANCH = "main"
CACHE_ROOT = Path(os.getenv("LANGUAGE_CLASSIFIER_CACHE", ".language_classifier_cache_v2"))
MAX_LENGTH = 128

MODEL_VARIANTS = {
    "4.3": {
        "kind": "zip",
        "repo": "abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents",
        "branch": "version4.1",
        "path": "models/bert_tiny_e2_multilingual_version4.3.zip",
        "url": "https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents/raw/refs/heads/version4.1/models/bert_tiny_e2_multilingual_version4.3.zip",
    },
    "New E2": {
        "kind": "zip",
        "branch": "main",
        "path": "artifacts/models/bert_tiny_e2_multilingual.zip",
    },
    "Version 4.1": {
        "kind": "zip",
        "repo": "abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents",
        "branch": "version4.1",
        "path": "models/bert_tiny_e2_multilingual_version4.1.zip",
    },
    "Version 4 (New E2)": {
        "kind": "zip",
        "branch": "version4",
        "path": "models/bert_tiny_e2_multilingual_version4.zip",
    },
}
DEFAULT_VARIANT = os.getenv("LANGUAGE_CLASSIFIER_VARIANT", "4.3")
_REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json")
_MODELS = {}


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=240, stream=True)
    response.raise_for_status()
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        if temporary.stat().st_size < 1024:
            raise ValueError(f"Downloaded file is unexpectedly small: {target.name}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_model_files(variant: str) -> Path:
    spec = MODEL_VARIANTS[variant]
    model_cache = CACHE_ROOT / variant.lower().replace(" ", "_")
    if spec["kind"] == "directory":
        model_cache.mkdir(parents=True, exist_ok=True)
        base_url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['branch']}/{spec['path']}"
        optional_files = ("tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
        for filename in (*_REQUIRED_FILES, *optional_files):
            target = model_cache / filename
            if target.exists() and target.stat().st_size:
                continue
            response = requests.get(f"{base_url}/{filename}", timeout=180)
            if response.status_code == 404 and filename in optional_files:
                continue
            response.raise_for_status()
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(target)
    else:
        archive = model_cache / Path(spec["path"]).name
        if not archive.exists() or archive.stat().st_size < 1024:
            source_repo = spec.get("repo", CLASSIFIER_REPO)
            url = spec.get("url") or (
                f"https://media.githubusercontent.com/media/"
                f"{source_repo}/{spec['branch']}/{spec['path']}"
            )
            try:
                _download(url, archive)
            except requests.HTTPError as first_error:
                # GitHub's media host can return 404 for this ZIP even when the
                # standard raw download URL works. Retry the known working URL.
                fallback_url = spec.get("url")
                if not fallback_url or fallback_url == url:
                    raise
                try:
                    _download(fallback_url, archive)
                except Exception:
                    raise first_error
        extract_dir = model_cache / "extracted"
        config_matches = list(extract_dir.rglob("config.json")) if extract_dir.exists() else []
        if not config_matches:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                root = extract_dir.resolve()
                for member in zf.infolist():
                    destination = (extract_dir / member.filename).resolve()
                    if destination != root and root not in destination.parents:
                        raise ValueError("Unsafe path found in model ZIP")
                zf.extractall(extract_dir)
            config_matches = list(extract_dir.rglob("config.json"))
        if not config_matches:
            raise FileNotFoundError(f"No config.json found in {archive.name}")
        model_cache = config_matches[0].parent

    for filename in _REQUIRED_FILES:
        if not (model_cache / filename).is_file():
            raise FileNotFoundError(f"Required model file missing for {variant}: {filename}")
    return model_cache


def _load(variant: str):
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown classifier variant {variant!r}. Choose from {list(MODEL_VARIANTS)}")
    if variant not in _MODELS:
        path = _ensure_model_files(variant)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        labels = {
            str(model.config.id2label.get(i, model.config.id2label.get(str(i), ""))).lower()
            for i in range(model.config.num_labels)
        }
        expected = {"as-in", "bn-in", "en-in", "gu-in", "hi-in", "kn-in", "ml-in", "mr-in", "none", "pa-in", "ta-in", "te-in"}
        if labels != expected:
            raise ValueError(f"Unexpected labels in {variant}: {labels}")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _MODELS[variant] = (tokenizer, model, device)
    return _MODELS[variant]


def classify_transcript(text: str, model_variant: str | None = None) -> dict:
    """Classify transcript with the selected multilingual BERT-Tiny model variant."""
    cleaned = " ".join(unicodedata.normalize("NFC", str(text or "")).split())
    if not cleaned:
        return {
            "label": "unavailable", "confidence": None, "probabilities": {},
            "latency_ms": 0.0, "available": False, "reason": "empty_text",
            "model_variant": model_variant or DEFAULT_VARIANT,
        }
    variant = model_variant or DEFAULT_VARIANT
    try:
        tokenizer, model, device = _load(variant)
        encoded = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            probs_tensor = torch.softmax(model(**encoded).logits, dim=-1)[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        values = probs_tensor.detach().cpu().tolist()
        id2label = model.config.id2label
        scores = {
            str(id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))): float(p)
            for i, p in enumerate(values)
        }
        label = max(scores, key=scores.get)
        return {
            "label": label, "confidence": scores[label], "probabilities": scores,
            "latency_ms": round(latency_ms, 3), "available": True, "reason": "",
            "model_variant": variant,
        }
    except Exception as exc:
        return {
            "label": "unavailable", "confidence": None, "probabilities": {},
            "latency_ms": 0.0, "available": False,
            "reason": f"{type(exc).__name__}: {exc}", "model_variant": variant,
        }
