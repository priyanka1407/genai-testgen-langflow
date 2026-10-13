"""Load settings from a .env file (KEY=VALUE lines) into the environment.

Variables already set in the shell win, and empty values are skipped, so a blank
`ANTHROPIC_API_KEY=` line never hides a real key set elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> list[str]:
    """Returns the names that were loaded (never the values)."""
    env = Path(path)
    if not env.is_file():
        return []
    loaded = []
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip('"').strip("'")
        if value and not os.getenv(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded
