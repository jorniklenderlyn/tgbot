#!/usr/bin/env python3
"""
Build the `tg_style` collection: pairs of (incoming message → user's reply)
for use as few-shot style examples when the assistant generates replies.

Usage:
  python scripts/build_style_index.py chat_*.json
  python scripts/build_style_index.py chat_*.json --reset
"""

import argparse
import glob
import json
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv
from qdrant_client.http import models as qm

load_dotenv()

from _rag import (
    embed_texts,
    ensure_collection,
    get_embedder,
    get_qdrant,
    message_payload_text,
)

TELEGRAM_USER_ID = int(os.environ["TELEGRAM_USER_ID"])
STYLE_COLLECTION = os.environ.get("STYLE_COLLECTION", "tg_style")

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


def main():
    parser = argparse.ArgumentParser(description="Build (incoming → user-reply) style index")
    parser.add_argument("json_files", nargs="+", help="Parsed chat JSON files (globs OK)")
    parser.add_argument("--reset", action="store_true", help="Drop existing style collection before ingest")
    args = parser.parse_args()

    files = []
    for pattern in args.json_files:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        print("No matching JSON files.", file=sys.stderr)
        sys.exit(1)

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
        print(f"{file}: {len(messages)} msgs → {len(pairs)} style pairs")

        if not pairs:
            continue

        vectors = embed_texts(embedder, [p["incoming_text"] for p in pairs])

        for pair, vec in zip(pairs, vectors):
            points.append(qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "chat_id": chat_id,
                    "chat_title": chat_title,
                    "incoming_text": pair["incoming_text"],
                    "reply_text": pair["reply_text"],
                    "incoming_id": pair["incoming_id"],
                    "reply_id": pair["reply_id"],
                    "incoming_date": pair["incoming_date"],
                    "reply_date": pair["reply_date"],
                },
            ))
            total_pairs += 1

            if len(points) >= 128:
                qdrant.upsert(collection_name=STYLE_COLLECTION, points=points)
                points = []

    if points:
        qdrant.upsert(collection_name=STYLE_COLLECTION, points=points)

    info = qdrant.get_collection(STYLE_COLLECTION)
    print(f"\nDone. Collection '{STYLE_COLLECTION}' now has {info.points_count} points "
          f"(added {total_pairs} in this run).")


if __name__ == "__main__":
    main()
