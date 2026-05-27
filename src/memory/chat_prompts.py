"""Load per-chat markdown prompts from CHAT_PROMPTS_DIR."""

import sys
from pathlib import Path

from src.config import CHAT_PROMPTS_DIR


def load_chat_prompts() -> dict[str, str]:
    """Return {key → markdown}. Keys: '@username' (lower), str(chat_id), '_default'."""
    out: dict[str, str] = {}
    d = Path(CHAT_PROMPTS_DIR)
    if not d.exists():
        return out
    for f in d.glob("*.md"):
        stem = f.stem
        if stem == "_example":
            continue
        key = stem.lower() if (stem.startswith("@") or stem == "_default") else stem
        try:
            out[key] = f.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[warn] failed to read {f}: {e}", file=sys.stderr)
    return out


def resolve_chat_prompt(prompts: dict[str, str], chat_id: int, username: str | None) -> str | None:
    if username:
        key = f"@{username.lower()}"
        if key in prompts:
            return prompts[key]
    if str(chat_id) in prompts:
        return prompts[str(chat_id)]
    return prompts.get("_default")
