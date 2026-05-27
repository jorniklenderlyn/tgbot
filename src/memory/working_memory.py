"""Per-chat working memory: persistent LLM-summarised conversation context.

Each chat gets a markdown file in .chat_contexts/{chat_id}.md that is
loaded before reply generation and updated asynchronously after each exchange.
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from src.config import ASSISTANT_LLM_MODEL, WORKING_MEMORY_DIR, WORKING_MEMORY_MAX_TOKENS
from src.util import load_prompt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MEMORY_DIR = _PROJECT_ROOT / WORKING_MEMORY_DIR

_chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

INIT_PROMPT = load_prompt("summarize_memory_init")
UPDATE_PROMPT_TEMPLATE = load_prompt("summarize_memory_update")

CHARS_PER_TOKEN = 4  # rough estimate for Russian text


def memory_path(chat_id: int) -> Path:
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return _MEMORY_DIR / f"{chat_id}.md"


def load_memory(chat_id: int) -> str | None:
    p = memory_path(chat_id)
    if p.exists():
        text = p.read_text(encoding="utf-8").strip()
        return text or None
    return None


def save_memory(chat_id: int, content: str) -> None:
    p = memory_path(chat_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)


async def build_initial_memory(
    telethon_client,
    chat_id: int,
    my_id: int,
    llm: OpenAI,
    model: str | None = None,
) -> str:
    """Fetch recent history from Telethon and summarise into initial context."""
    model = model or ASSISTANT_LLM_MODEL
    max_chars = WORKING_MEMORY_MAX_TOKENS * CHARS_PER_TOKEN

    lines: list[str] = []
    total_chars = 0

    async for msg in telethon_client.iter_messages(chat_id, limit=200):
        text = (msg.message or "").strip()
        if not text:
            continue
        prefix = "Я" if msg.sender_id == my_id else "Собеседник"
        line = f"{prefix}: {text}"
        total_chars += len(line)
        lines.append(line)
        if total_chars >= max_chars:
            break

    if not lines:
        return ""

    lines.reverse()
    transcript = "\n".join(lines)

    prompt = INIT_PROMPT + "\n" + transcript

    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[memory] initial build failed for {chat_id}: {e}", file=sys.stderr)
        return ""


async def update_memory(
    chat_id: int,
    current_memory: str | None,
    new_exchange: str,
    llm: OpenAI,
    model: str | None = None,
) -> None:
    """Incrementally update the memory file. Acquires per-chat lock."""
    model = model or ASSISTANT_LLM_MODEL

    async with _chat_locks[chat_id]:
        fresh_memory = load_memory(chat_id)
        mem = fresh_memory or current_memory or ""

        prompt = UPDATE_PROMPT_TEMPLATE.replace(
            "{current_memory}", mem,
        ).replace(
            "{new_exchange}", new_exchange,
        )

        try:
            resp = llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            updated = (resp.choices[0].message.content or "").strip()
            if updated:
                save_memory(chat_id, updated)
        except Exception as e:
            print(f"[memory] update failed for {chat_id}: {e}", file=sys.stderr)
