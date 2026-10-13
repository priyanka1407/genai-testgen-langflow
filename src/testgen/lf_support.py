"""Helpers shared by the Langflow components (kept in the package so every
component file stays self-contained for Langflow's loader)."""
from __future__ import annotations

from .env import load_env_file
from .llm import get_llm

# Langflow started without start-langflow.ps1 still picks up keys from the project's .env
load_env_file()

# The provider follows the model name: gemini-* -> Gemini, claude-* -> Claude.
# The dropdown also accepts a typed model id (combobox) for models not listed here.
MODELS = [
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
    "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite",
]
API_KEY_INFO = ("Anthropic key for claude-* models, Gemini key for gemini-* models. Leave empty to use "
                "ANTHROPIC_API_KEY / GEMINI_API_KEY from the environment")


def llm_from(component):
    return get_llm(offline=component.offline, api_key=component.api_key or None, model=component.model)


def as_dicts(items) -> list[dict]:
    """Langflow passes Data objects (or a single one); normalise to plain dicts."""
    items = items if isinstance(items, list) else [items]
    return [i.data if hasattr(i, "data") else i for i in items if i is not None]
