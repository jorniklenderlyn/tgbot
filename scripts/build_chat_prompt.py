#!/usr/bin/env python3
"""
Draft a per-chat prompt by analysing parsed chat history with an LLM.

Reads a parsed chat JSON (output of parse_chat.py), feeds a sampled slice
to the LLM, and writes a markdown profile to chat_prompts/<key>.md.

Usage:
  python scripts/build_chat_prompt.py @friend1 --json chat_friend1_*.json
  python scripts/build_chat_prompt.py 12345 --json chat.json --force
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from _rag import render_message  # noqa: E402
from src.prompt_loader import load_prompt  # noqa: E402

TELEGRAM_USER_ID = int(os.environ["TELEGRAM_USER_ID"])
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
CHAT_PROMPTS_DIR = os.environ.get("CHAT_PROMPTS_DIR", "chat_prompts")

SAMPLE_RECENT = 200
SAMPLE_HEAD = 100
SAMPLE_MIDDLE = 100

PROMPT = load_prompt("build_chat_profile")


def is_user(msg: dict) -> bool:
    return (msg.get("sender") or {}).get("id") == TELEGRAM_USER_ID


def render_for_prompt(msg: dict) -> str:
    sender = msg.get("sender") or {}
    name = "Я" if is_user(msg) else (sender.get("name") or sender.get("username") or "Он")
    text_parts = []
    if msg.get("text"):
        text_parts.append(msg["text"])
    if msg.get("audio_transcript"):
        text_parts.append(f"[голос/видео: {msg['audio_transcript']}]")
    if msg.get("image_description"):
        text_parts.append(f"[фото: {msg['image_description']}]")
    text = " ".join(text_parts).strip()
    if not text:
        return ""
    return f"{name}: {text}"


def sample_messages(messages: list[dict]) -> list[str]:
    rendered = [r for r in (render_for_prompt(m) for m in messages) if r]
    if len(rendered) <= SAMPLE_RECENT + SAMPLE_HEAD + SAMPLE_MIDDLE:
        return rendered
    head = rendered[:SAMPLE_HEAD]
    mid_start = (len(rendered) - SAMPLE_MIDDLE) // 2
    middle = rendered[mid_start:mid_start + SAMPLE_MIDDLE]
    tail = rendered[-SAMPLE_RECENT:]
    return head + ["..."] + middle + ["..."] + tail


def output_filename(key: str) -> Path:
    safe = key
    if safe.startswith("@"):
        safe = safe.lower()
    return Path(CHAT_PROMPTS_DIR) / f"{safe}.md"


def main():
    parser = argparse.ArgumentParser(description="Draft a per-chat prompt MD file")
    parser.add_argument("key", help="Output key: @username or numeric chat_id (becomes the filename)")
    parser.add_argument("--json", required=True, help="Parsed chat JSON (glob OK, latest used)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    files = sorted(glob.glob(args.json))
    if not files:
        print(f"No matching JSON: {args.json}", file=sys.stderr)
        sys.exit(1)
    src = files[-1]

    out_path = output_filename(args.key)
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    messages = data["messages"]
    sample = sample_messages(messages)
    if not sample:
        print("No usable messages in this JSON.", file=sys.stderr)
        sys.exit(1)

    print(f"Using {len(sample)} sampled lines from {src}", file=sys.stderr)

    llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    transcript = "\n".join(sample)

    print(f"Calling {OPENROUTER_LLM_MODEL}...", file=sys.stderr)
    resp = llm.chat.completions.create(
        model=OPENROUTER_LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT + transcript}],
        temperature=0.3,
    )
    md = resp.choices[0].message.content.strip()
    if md.startswith("```"):
        md = md.split("```", 2)[1]
        if md.startswith("markdown"):
            md = md[len("markdown"):]
        md = md.strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md + "\n", encoding="utf-8")
    print(f"\nWrote draft to {out_path}")
    print("→ Review and edit the file before activating this chat in whitelist.json.")


if __name__ == "__main__":
    main()
