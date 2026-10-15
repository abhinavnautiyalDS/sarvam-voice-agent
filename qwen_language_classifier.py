"""Optional Qwen LoRA LLM transcript classifier.

The model is loaded only when the user selects the Qwen LLM backend. The LoRA
adapter is downloaded from the companion public repository; the Qwen base model
is fetched from Hugging Face by Transformers. This classifier does not expose a
calibrated confidence score, so confidence is returned as None.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

BASE_MODEL = os.getenv("HINGLISH_LLM_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = Path(
    os.getenv(
        "HINGLISH_LLM_ADAPTER_PATH",
        str(Path(__file__).parent / "models" / "llm"),
    )
)
ADAPTER_REPO_RAW = os.getenv(
    "HINGLISH_LLM_ADAPTER_RAW_BASE",
    "https://raw.githubusercontent.com/abhinavnautiyalDS/hinglish-language-switch-llm/main/models/qwen-hinglish-lora",
)
_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
_LABELS = ("HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN")
_INSTRUCTION = (
    "Classify the caller utterance for a voice-agent language switch. "
    "HINGLISH means Hindi in Devanagari or Latin script, including Hindi-English code-mixing. "
    "ENGLISH means meaningful English. NEUTRAL means a generic greeting/acknowledgement "
    "or too little language evidence to trigger a switch. UNCERTAIN means unsupported "
    "language or unclear/ambiguous content. Return exactly one label and no explanation."
)

_MODEL: Any | None = None
_TOKENIZER: Any | None = None
_DEVICE = "cpu"
_LOAD_ERROR = ""
_LOCK = threading.Lock()


def _ensure_adapter_files() -> None:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    for filename in _REQUIRED_FILES:
        destination = ADAPTER_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        url = f"{ADAPTER_REPO_RAW.rstrip('/')}/{filename}"
        response = httpx.get(url, timeout=180, follow_redirects=True)
        response.raise_for_status()
        if len(response.content) < 100:
            raise ValueError(f"Downloaded adapter file looks too small: {filename}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(destination)


def load_model() -> tuple[Any | None, Any | None, str]:
    """Load the base Qwen model and LoRA adapter once, on first use."""
    global _MODEL, _TOKENIZER, _DEVICE, _LOAD_ERROR
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER, ""
    with _LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER, ""
        try:
            _ensure_adapter_files()
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
            model.to(_DEVICE).eval()
            _TOKENIZER = tokenizer
            _MODEL = model
            _LOAD_ERROR = ""
        except Exception as exc:
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            _MODEL = None
            _TOKENIZER = None
    return _MODEL, _TOKENIZER, _LOAD_ERROR


def classify_transcript(text: str) -> dict[str, Any]:
    """Classify one transcript and report per-utterance inference latency."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return {
            "label": "uncertain", "confidence": None, "probabilities": {},
            "latency_ms": 0.0, "available": True, "reason": "empty_transcript",
        }

    model, tokenizer, error = load_model()
    if model is None or tokenizer is None:
        return {
            "label": "unavailable", "confidence": None, "probabilities": {},
            "latency_ms": 0.0, "available": False,
            "reason": error or "qwen_model_not_loaded",
        }

    messages = [
        {"role": "system", "content": _INSTRUCTION},
        {"role": "user", "content": clean},
    ]
    try:
        import torch

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs = {"input_ids": encoded["input_ids"].to(_DEVICE)}
        if "attention_mask" in encoded:
            model_inputs["attention_mask"] = encoded["attention_mask"].to(_DEVICE)
        input_ids = model_inputs["input_ids"]
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **model_inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        answer = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip().upper()
        match = re.search(r"\b(HINGLISH|ENGLISH|NEUTRAL|UNCERTAIN)\b", answer)
        label = match.group(1).lower() if match else "uncertain"
        return {
            "label": label,
            "confidence": None,
            "probabilities": {},
            "latency_ms": latency_ms,
            "available": True,
            "reason": "generative_classifier_confidence_not_calibrated",
        }
    except Exception as exc:
        return {
            "label": "unavailable", "confidence": None, "probabilities": {},
            "latency_ms": 0.0, "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
