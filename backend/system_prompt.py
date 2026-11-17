"""Load only the selected AURA agent's concise system prompt."""
from pathlib import Path

AGENT_PROMPTS = {
    "general": "prompts/general.txt",
    "customer_service": "prompts/customer_service.txt",
    "tutor": "prompts/tutor.txt",
}


def get_system_prompt(agent_type: str = "general") -> str:
    """Return one prompt file only; never combine prompts across agents."""
    relative_path = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["general"])
    prompt_path = Path(__file__).resolve().parent / relative_path
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"System prompt unavailable for selected agent: {agent_type}") from exc
    if not prompt:
        raise RuntimeError(f"System prompt is empty for selected agent: {agent_type}")
    return prompt
