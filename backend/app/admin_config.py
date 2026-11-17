from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(os.getenv("AURA_ADMIN_CONFIG_PATH", "data/admin_config.json"))


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(updates: dict[str, Any]) -> None:
    current = load_config()
    for key, value in updates.items():
        if value is None or value == "":
            current.pop(key, None)
        else:
            current[key] = value
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".aura-config-", dir=str(CONFIG_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2)
            handle.write("\n")
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, CONFIG_PATH)
        os.chmod(CONFIG_PATH, 0o600)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def get_setting(name: str, default: str = "") -> str:
    config = load_config()
    value = config.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.getenv(name, default).strip()


def public_config() -> dict[str, Any]:
    config = load_config()
    def configured(name: str) -> bool:
        return bool(config.get(name) or os.getenv(name))
    return {
        "sarvam_api_key_set": configured("SARVAM_API_KEY"),
        "bakbak_api_key_set": configured("BAKBAK_API_KEY"),
        "bakbak_voice_id_set": configured("BAKBAK_VOICE_ID"),
        "sarvam_llm_model": get_setting("SARVAM_LLM_MODEL", "sarvam-105b-conversations"),
        "bakbak_tts_model": get_setting("BAKBAK_TTS_MODEL", "m1"),
        "sarvam_stt_mode": get_setting("SARVAM_STT_MODE", "verbose"),
        "llm_provider": get_setting("AURA_LLM_PROVIDER", "sarvam"),
        "tts_provider": get_setting("AURA_TTS_PROVIDER", "bakbak"),
    }
