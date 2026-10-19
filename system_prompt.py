"""Load the repository's canonical voice-agent system prompt."""
import os
from pathlib import Path


def get_system_prompt() -> str:
    """Prefer the checked-in prompt; use the environment variable only as fallback."""
    prompt_path = Path(__file__).with_name("system_prompt.txt")
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError:
        prompt = ""
    if prompt:
        return prompt
    return os.getenv(
        "VOICE_SYSTEM_PROMPT",
        "You are a concise, natural voice assistant. Answer directly. Avoid markdown, bullets, and emojis.",
    )
