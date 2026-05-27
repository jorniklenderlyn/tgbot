#!/usr/bin/env python3
"""
Ingest a parsed-chat JSON into Qdrant for RAG.

Pipeline:
  1. Load messages from JSON produced by parse_chat.py
  2. Render each message into a single text "block" (text + transcript + image desc + sender + time + reply context)
  3. Group consecutive messages into conversation chunks (by time gap / size)
  4. Use LLM to extract facts and events from each chunk
  5. Embed every chunk + fact + event with a multilingual model
  6. Upsert into Qdrant
"""

import argparse
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
    render_message,
    embed_texts,
    get_embedder,
    get_qdrant,
    ensure_collection,
)
from src.util.prompt_loader import load_prompt

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "tg_chat")

CHUNK_TIME_GAP_MINUTES = 30
CHUNK_MAX_MESSAGES = 30

FACT_EXTRACTION_PROMPT = load_prompt("extract_facts_events")


def chunk_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        return []

    chunks = []
    current = [messages[0]]
    last_time = datetime.fromisoformat(messages[0]["date"])

    for msg in messages[1:]:
        msg_time = datetime.fromisoformat(msg["date"])
        gap_minutes = (msg_time - last_time).total_seconds() / 60

        if gap_minutes > CHUNK_TIME_GAP_MINUTES or len(current) >= CHUNK_MAX_MESSAGES:
            chunks.append(_build_chunk(current))
            current = [msg]
        else:
            current.append(msg)
        last_time = msg_time

    if current:
        chunks.append(_build_chunk(current))

    return chunks


def _build_chunk(msgs: list[dict]) -> dict:
    rendered = "\n".join(render_message(m) for m in msgs)
    participants = sorted({
        m.get("sender", {}).get("name") or str(m.get("sender", {}).get("id", "?"))
        for m in msgs
    })
    return {
        "text": rendered,
        "date_from": msgs[0]["date"],
        "date_to": msgs[-1]["date"],
        "message_ids": [m["id"] for m in msgs],
        "participants": participants,
    }


def extract_facts_and_events(llm: OpenAI, chunk_text: str) -> dict:
    try:
        resp = llm.chat.completions.create(
            model=OPENROUTER_LLM_MODEL,
            messages=[{"role": "user", "content": FACT_EXTRACTION_PROMPT + chunk_text}],
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"  [warn] Extraction failed: {e}", file=sys.stderr)
        return {"facts": [], "events": []}


def main():
    parser = argparse.ArgumentParser(description="Ingest parsed Telegram chat JSON into Qdrant")
    parser.add_argument("json_file", help="Path to parsed chat JSON")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip LLM fact/event extraction (much faster, only stores messages)")
    parser.add_argument("--reset", action="store_true", help="Drop existing collection before ingesting")
    args = parser.parse_args()

    with open(args.json_file, encoding="utf-8") as f:
        data = json.load(f)

    chat_id = str(data["chat"]["id"])
    chat_title = data["chat"]["title"]
    messages = data["messages"]
    print(f"Loaded {len(messages)} messages from chat '{chat_title}'")

    print("Building conversation chunks...")
    chunks = chunk_messages(messages)
    print(f"  → {len(chunks)} chunks")

    embedder = get_embedder()
    vector_size = embedder.get_sentence_embedding_dimension()

    qdrant = get_qdrant()
    if args.reset:
        try:
            qdrant.delete_collection(QDRANT_COLLECTION)
            print(f"Dropped collection '{QDRANT_COLLECTION}'")
        except Exception:
            pass
    ensure_collection(qdrant, QDRANT_COLLECTION, vector_size,
                      indexed_fields=["chat_id", "type", "date_from", "participants"])

    llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    points = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Processing chunk {i}/{len(chunks)} ({len(chunk['message_ids'])} msgs)...")
        chunk_vec = embed_texts(embedder, [chunk["text"]])[0]
        points.append(qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk_vec,
            payload={
                "chat_id": chat_id,
                "chat_title": chat_title,
                "type": "message_chunk",
                "text": chunk["text"],
                "date_from": chunk["date_from"],
                "date_to": chunk["date_to"],
                "message_ids": chunk["message_ids"],
                "participants": chunk["participants"],
            },
        ))

        if args.skip_extraction:
            continue

        extracted = extract_facts_and_events(llm, chunk["text"])
        for fact in extracted.get("facts", []):
            text = fact.get("text", "").strip()
            if not text:
                continue
            vec = embed_texts(embedder, [text])[0]
            points.append(qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "chat_id": chat_id,
                    "chat_title": chat_title,
                    "type": "fact",
                    "text": text,
                    "participants": fact.get("participants", []),
                    "source_date_from": chunk["date_from"],
                    "source_date_to": chunk["date_to"],
                },
            ))
        for event in extracted.get("events", []):
            text = event.get("text", "").strip()
            if not text:
                continue
            vec = embed_texts(embedder, [text])[0]
            points.append(qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "chat_id": chat_id,
                    "chat_title": chat_title,
                    "type": "event",
                    "text": text,
                    "date": event.get("date"),
                    "participants": event.get("participants", []),
                    "source_date_from": chunk["date_from"],
                    "source_date_to": chunk["date_to"],
                },
            ))

        if len(points) >= 64:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            points = []

    if points:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

    info = qdrant.get_collection(QDRANT_COLLECTION)
    print(f"\nDone. Collection '{QDRANT_COLLECTION}' now has {info.points_count} points.")


if __name__ == "__main__":
    main()
