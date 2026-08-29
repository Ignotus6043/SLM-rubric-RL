from __future__ import annotations

import os
from pathlib import Path

ENV_ALIASES = {
    "OPENAI-KEY": "OPENAI_API_KEY",
    "OPENAI_ORG_ID": "OPENAI_ORG_ID",
    "OPENAI-ORG-ID": "OPENAI_ORG_ID",
    "WANDB-API-KEY": "WANDB_API_KEY",
}


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip().lstrip("\ufeff")
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip().lstrip("\ufeff")
    value = value.strip()

    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    return key, value


def load_dotenv_if_present(dotenv_path: str | None) -> Path | None:
    """Load KEY=VALUE pairs from a .env file into os.environ if present."""
    if not dotenv_path:
        return None

    path = Path(dotenv_path).expanduser()
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not os.environ.get(key):
            os.environ[key] = value
        alias_key = ENV_ALIASES.get(key)
        if alias_key and not os.environ.get(alias_key):
            os.environ[alias_key] = value

    return path
