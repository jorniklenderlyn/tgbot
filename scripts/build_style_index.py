#!/usr/bin/env python3
"""
Build the `tg_style` collection: a reusable STYLE bank of (incoming -> reply)
pairs for few-shot style imitation.

By default each pair is passed through an LLM that:
  - removes specific facts (names, times, topics ...) -> placeholders, so an
    example reused in a DIFFERENT chat can't leak private data, and
  - attaches a fact-free `situation` label describing the context, so callers
    can judge whether an example fits a new case.
The embedding is computed on the redacted incoming text; no raw fact-bearing
text is stored. Use --no-redact for the old behaviour (store raw text).

Usage:
  python scripts/build_style_index.py chat_*.json --reset
  python scripts/build_style_index.py chat_*.json --no-redact      # legacy raw
  python scripts/build_style_index.py chat_*.json --batch-size 20
"""

import argparse
import glob
import json
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.http import models as qm

load_dotenv()

from _rag import (
    embed_texts,
    ensure_collection,
    get_embedder,
    get_qdrant,
    message_payload_text,
)
from src.util.prompt_loader import load_prompt

TELEGRAM_USER_ID = int(os.environ["TELEGRAM_USER_ID"])
STYLE_COLLECTION = os.environ.get("STYLE_COLLECTION", "tg_style")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
STYLE_LLM_MODEL = os.environ.get("STYLE_LLM_MODEL", "deepseek/deepseek-v4-flash")
STYLE_LLM_PROVIDER = os.environ.get("STYLE_LLM_PROVIDER", "DeepInfra")

PAIR_MAX_GAP_MINUTES = 60


def is_user_message(msg: dict) -> bool:
    sender = msg.get("sender") or {}
    return sender.get("id") == TELEGRAM_USER_ID


def find_pairs(messages: list[dict]) -> list[dict]:
    """Yield (incoming, reply) pairs where `reply` is authored by TELEGRAM_USER_ID."""
    by_id = {m["id"]: m for m in messages}
    pairs = []

    for i, msg in enumerate(messages):
        if not is_user_message(msg):
            continue

        reply_text = message_payload_text(msg)
        if not reply_text:
            continue

        incoming = None

        # Explicit reply-to
        if msg.get("is_reply") and msg.get("reply_to_msg_id"):
            target = by_id.get(msg["reply_to_msg_id"])
            if target and not is_user_message(target):
                incoming = target

        # Otherwise: walk back to the most recent non-user message within gap
        if incoming is None:
            my_time = datetime.fromisoformat(msg["date"])
            for j in range(i - 1, -1, -1):
                prev = messages[j]
                if is_user_message(prev):
                    continue
                prev_time = datetime.fromisoformat(prev["date"])
                gap_min = (my_time - prev_time).total_seconds() / 60
                if gap_min > PAIR_MAX_GAP_MINUTES:
                    break
                if message_payload_text(prev):
                    incoming = prev
                    break

        if incoming is None:
            continue

        incoming_text = message_payload_text(incoming)
        if not incoming_text:
            continue

        pairs.append({
            "incoming_text": incoming_text,
            "incoming_id": incoming["id"],
            "incoming_date": incoming["date"],
            "reply_text": reply_text,
            "reply_id": msg["id"],
            "reply_date": msg["date"],
        })

    return pairs


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip()


def redact_pairs(llm: OpenAI, pairs: list[dict], batch_size: int) -> list[dict]:
    """Add situation + redacted incoming/reply to each pair (facts removed).

    Pairs that fail redaction are dropped (never stored with raw facts).
    """
    prompt = load_prompt("redact_style_pair")
    extra = {"usage": {"include": True}}
    if STYLE_LLM_PROVIDER:
        extra["provider"] = {"order": [STYLE_LLM_PROVIDER], "allow_fallbacks": False}

    out: list[dict] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        payload = [{"id": i, "incoming": p["incoming_text"], "reply": p["reply_text"]}
                   for i, p in enumerate(batch)]
        try:
            resp = llm.chat.completions.create(
                model=STYLE_LLM_MODEL,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0, extra_body=extra)
            items = json.loads(_strip_fences(resp.choices[0].message.content))
            by_id = {it["id"]: it for it in items if isinstance(it, dict) and "id" in it}
        except Exception as e:
            print(f"  [redact] batch {start}-{start+len(batch)} failed: {e}; skipped",
                  file=sys.stderr)
            continue

        for i, p in enumerate(batch):
            it = by_id.get(i)
            if not it or not it.get("reply_redacted"):
                continue
            q = dict(p)
            q["situation"] = (it.get("situation") or "").strip()
            q["incoming_text"] = (it.get("incoming_redacted") or p["incoming_text"]).strip()
            q["reply_text"] = (it.get("reply_redacted") or "").strip()
            out.append(q)
        print(f"  [redact] {min(start + batch_size, len(pairs))}/{len(pairs)} pairs",
              file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build (incoming -> user-reply) style index")
    parser.add_argument("json_files", nargs="+", help="Parsed chat JSON files (globs OK)")
    parser.add_argument("--reset", action="store_true", help="Drop existing style collection before ingest")
    parser.add_argument("--no-redact", action="store_true",
                        help="Store raw text instead of fact-redacted style (legacy)")
    parser.add_argument("--batch-size", type=int, default=15, help="Pairs per redaction LLM call")
    args = parser.parse_args()

    files = []
    for pattern in args.json_files:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        print("No matching JSON files.", file=sys.stderr)
        sys.exit(1)

    redact = not args.no_redact
    if redact and not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set — needed for redaction. Use --no-redact to skip.",
              file=sys.stderr)
        sys.exit(1)
    llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1") if redact else None

    embedder = get_embedder()
    vector_size = embedder.get_sentence_embedding_dimension()

    qdrant = get_qdrant()
    if args.reset:
        try:
            qdrant.delete_collection(STYLE_COLLECTION)
            print(f"Dropped collection '{STYLE_COLLECTION}'")
        except Exception:
            pass
    ensure_collection(qdrant, STYLE_COLLECTION, vector_size,
                      indexed_fields=["chat_id", "chat_title"])

    total_pairs = 0
    points = []

    for file in files:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        chat_id = str(data["chat"]["id"])
        chat_title = data["chat"]["title"]
        messages = data["messages"]
        pairs = find_pairs(messages)
        print(f"{file}: {len(messages)} msgs → {len(pairs)} style pairs", file=sys.stderr)

        if not pairs:
            continue

        if redact:
            print(f"  redacting facts + tagging context via {STYLE_LLM_MODEL}...", file=sys.stderr)
            pairs = redact_pairs(llm, pairs, args.batch_size)
            print(f"  → {len(pairs)} pairs kept after redaction", file=sys.stderr)
            if not pairs:
                continue

        # Embed on the (redacted) incoming text so query-time raw incomings match.
        vectors = embed_texts(embedder, [p["incoming_text"] for p in pairs])

        for pair, vec in zip(pairs, vectors):
            payload = {
                "chat_id": chat_id,
                "chat_title": chat_title,
                "incoming_text": pair["incoming_text"],
                "reply_text": pair["reply_text"],
                "incoming_id": pair["incoming_id"],
                "reply_id": pair["reply_id"],
                "incoming_date": pair["incoming_date"],
                "reply_date": pair["reply_date"],
                "redacted": redact,
            }
            if redact:
                payload["situation"] = pair.get("situation", "")
            points.append(qm.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
            total_pairs += 1

            if len(points) >= 128:
                qdrant.upsert(collection_name=STYLE_COLLECTION, points=points)
                points = []

    if points:
        qdrant.upsert(collection_name=STYLE_COLLECTION, points=points)

    info = qdrant.get_collection(STYLE_COLLECTION)
    print(f"\nDone. Collection '{STYLE_COLLECTION}' now has {info.points_count} points "
          f"(added {total_pairs} in this run; redacted={redact}).")


if __name__ == "__main__":
    main()
