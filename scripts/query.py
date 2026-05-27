#!/usr/bin/env python3
"""
Ask questions about your Telegram chat using the RAG vector DB.

Usage:
  python scripts/query.py "когда мы договаривались встретиться?"
  python scripts/query.py        # interactive mode
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager

import requests
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

load_dotenv()

from _rag import embed_one, get_embedder, get_qdrant
from src.util.prompt_loader import load_prompt

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "tg_chat")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL_DEFAULT = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = load_prompt("query_rag")


def search(qdrant: QdrantClient, vector: list[float], top_k: int = 10, type_filter: str | None = None) -> list:
    query_filter = None
    if type_filter:
        query_filter = qm.Filter(must=[qm.FieldCondition(key="type", match=qm.MatchValue(value=type_filter))])
    return qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter,
    )


def build_context(hits: list) -> str:
    blocks = []
    for h in hits:
        p = h.payload
        t = p.get("type")
        if t == "fact":
            blocks.append(f"[ФАКТ, score={h.score:.2f}] {p['text']}")
        elif t == "event":
            date = p.get("date") or p.get("source_date_from")
            blocks.append(f"[СОБЫТИЕ {date}, score={h.score:.2f}] {p['text']}")
        else:
            blocks.append(
                f"[ПЕРЕПИСКА {p.get('date_from')} — {p.get('date_to')}, score={h.score:.2f}]\n{p['text']}"
            )
    return "\n\n".join(blocks)


def answer_openrouter(question: str, context: str, model: str) -> str:
    llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"КОНТЕКСТ ИЗ ИСТОРИИ ЧАТА:\n\n{context}\n\nВОПРОС: {question}"},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def answer_ollama(question: str, context: str, model: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"КОНТЕКСТ ИЗ ИСТОРИИ ЧАТА:\n\n{context}\n\nВОПРОС: {question}"},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def answer(question: str, context: str, backend: str, model: str) -> str:
    if backend == "ollama":
        return answer_ollama(question, context, model)
    return answer_openrouter(question, context, model)


@contextmanager
def timed(label: str, verbose: bool):
    t0 = time.perf_counter()
    yield
    if verbose:
        dt = (time.perf_counter() - t0) * 1000
        print(f"  ⏱  {label}: {dt:.0f} ms", file=sys.stderr)


TYPE_LABELS = {"fact": "ФАКТ", "event": "СОБЫТИЕ", "message_chunk": "ПЕРЕПИСКА"}


def format_hits_for_display(facts: list, events: list, chunks: list) -> str:
    """Render retrieved hits as a human-readable inspection report."""
    sections = []

    def fmt_payload(h, body: str) -> str:
        return f"score={h.score:.3f}\n{body}"

    if facts:
        lines = [f"━━ FACTS ({len(facts)}) ━━"]
        for i, h in enumerate(facts, 1):
            p = h.payload
            lines.append(
                f"\n[{i}] {fmt_payload(h, p['text'])}\n"
                f"    участники: {', '.join(p.get('participants') or []) or '—'}\n"
                f"    источник: {p.get('source_date_from', '?')} … {p.get('source_date_to', '?')}"
            )
        sections.append("\n".join(lines))

    if events:
        lines = [f"━━ EVENTS ({len(events)}) ━━"]
        for i, h in enumerate(events, 1):
            p = h.payload
            date = p.get("date") or p.get("source_date_from")
            lines.append(
                f"\n[{i}] {fmt_payload(h, p['text'])}\n"
                f"    дата: {date}\n"
                f"    участники: {', '.join(p.get('participants') or []) or '—'}"
            )
        sections.append("\n".join(lines))

    if chunks:
        lines = [f"━━ MESSAGE CHUNKS ({len(chunks)}) ━━"]
        for i, h in enumerate(chunks, 1):
            p = h.payload
            preview = p["text"]
            lines.append(
                f"\n[{i}] {fmt_payload(h, '')}\n"
                f"    {p.get('date_from', '?')} … {p.get('date_to', '?')}\n"
                f"    участники: {', '.join(p.get('participants') or []) or '—'}\n"
                f"    сообщений: {len(p.get('message_ids') or [])}\n"
                f"--- text ---\n{preview}\n--- end ---"
            )
        sections.append("\n".join(lines))

    if not sections:
        return "(no hits)"
    return "\n\n".join(sections)


def ask(qdrant: QdrantClient, embedder: SentenceTransformer, question: str,
        top_k: int, show_context: bool, backend: str, model: str, verbose: bool,
        retrieve_only: bool, types: set[str]):
    t_total = time.perf_counter()

    with timed("embed query", verbose):
        vec = embed_one(embedder, question, is_query=True)

    facts: list = []
    events: list = []
    chunks: list = []

    if "fact" in types:
        with timed(f"qdrant search facts (k={max(3, top_k // 3)})", verbose):
            facts = search(qdrant, vec, top_k=max(3, top_k // 3), type_filter="fact")
    if "event" in types:
        with timed(f"qdrant search events (k={max(3, top_k // 3)})", verbose):
            events = search(qdrant, vec, top_k=max(3, top_k // 3), type_filter="event")
    if "chunk" in types:
        with timed(f"qdrant search chunks (k={top_k})", verbose):
            chunks = search(qdrant, vec, top_k=top_k, type_filter="message_chunk")

    hits = facts + events + chunks
    if not hits:
        print("Ничего не найдено в базе.")
        return

    if verbose:
        print(f"  📊  hits: {len(facts)} facts, {len(events)} events, {len(chunks)} chunks",
              file=sys.stderr)

    if retrieve_only:
        print(format_hits_for_display(facts, events, chunks))
        if verbose:
            total_ms = (time.perf_counter() - t_total) * 1000
            print(f"\n  ⏱  TOTAL: {total_ms:.0f} ms", file=sys.stderr)
        return

    context = build_context(hits)
    if verbose:
        print(f"  📏  context size: {len(context)} chars", file=sys.stderr)

    if show_context:
        print("\n--- CONTEXT ---")
        print(context)
        print("--- END CONTEXT ---\n")

    print(f"Думаю ({backend}:{model})...", file=sys.stderr)
    with timed(f"LLM call ({backend}:{model})", verbose):
        response = answer(question, context, backend, model)

    if verbose:
        total_ms = (time.perf_counter() - t_total) * 1000
        print(f"  ⏱  TOTAL: {total_ms:.0f} ms", file=sys.stderr)

    print(f"\n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your Telegram chat")
    parser.add_argument("question", nargs="?", help="Question to ask (if omitted, interactive mode)")
    parser.add_argument("-k", "--top-k", type=int, default=8, help="Number of message chunks to retrieve (default 8)")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context before the answer")
    parser.add_argument("--llm", choices=["openrouter", "ollama"], default="openrouter",
                        help="LLM backend: openrouter (cloud) or ollama (local). Default: openrouter")
    parser.add_argument("--model", default=None,
                        help=f"Model name. Default: '{OPENROUTER_LLM_MODEL}' for openrouter, "
                             f"'{OLLAMA_MODEL_DEFAULT}' for ollama")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show timing for embedding, qdrant searches, and LLM call")
    parser.add_argument("--retrieve-only", "-r", action="store_true",
                        help="Show retrieved hits with scores and skip the LLM call entirely")
    parser.add_argument("--types", default="fact,event,chunk",
                        help="Comma-separated subset of hit types to retrieve: fact,event,chunk (default: all)")
    args = parser.parse_args()

    types = {t.strip() for t in args.types.split(",") if t.strip()}
    unknown = types - {"fact", "event", "chunk"}
    if unknown:
        print(f"Error: unknown types in --types: {unknown}. Allowed: fact, event, chunk",
              file=sys.stderr)
        sys.exit(1)

    backend = args.llm
    if args.model:
        model = args.model
    elif backend == "ollama":
        model = OLLAMA_MODEL_DEFAULT
    else:
        model = OPENROUTER_LLM_MODEL

    if not args.retrieve_only and backend == "openrouter" and not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY is not set in .env", file=sys.stderr)
        sys.exit(1)
    if not args.retrieve_only and backend == "ollama":
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            r.raise_for_status()
        except Exception as e:
            print(f"Error: ollama not reachable at {OLLAMA_URL} ({e}). "
                  f"Start it with 'brew services start ollama' and 'ollama pull {model}'.",
                  file=sys.stderr)
            sys.exit(1)

    embedder = get_embedder()
    qdrant = get_qdrant()

    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        print(f"Collection '{QDRANT_COLLECTION}': {info.points_count} points", file=sys.stderr)
    except Exception as e:
        print(f"Error: collection '{QDRANT_COLLECTION}' not found. Run ingest first. ({e})", file=sys.stderr)
        sys.exit(1)

    if args.question:
        ask(qdrant, embedder, args.question, args.top_k, args.show_context, backend, model,
            args.verbose, args.retrieve_only, types)
        return

    print(f"Interactive mode ({backend}:{model}). Type your question (or 'exit'):", file=sys.stderr)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        ask(qdrant, embedder, q, args.top_k, args.show_context, backend, model, args.verbose,
            args.retrieve_only, types)


if __name__ == "__main__":
    main()
