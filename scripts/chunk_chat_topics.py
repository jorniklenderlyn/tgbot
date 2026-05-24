#!/usr/bin/env python3
"""
Group a parsed chat dump into topic-coherent chunks for later style analysis.

Two-stage segmentation, fully offline (no API calls):
  1. Time-gap split — break the message stream into "sessions" wherever the
     gap between consecutive messages exceeds --gap-minutes. A long silence is
     a natural conversation boundary.
  2. Embedding split — within each session, run a TextTiling-style topic
     segmentation over the local sentence-transformers embeddings (the same
     multilingual-e5 model the rest of the project uses). Semantic dips
     between adjacent message blocks mark topic shifts.

The result is a JSON file of chunks, each a contiguous run of messages that
hang together on one topic, with light metadata (time span, participants,
counts) — no LLM labels.

Usage:
  python scripts/chunk_chat_topics.py chat_ntsupkov_20260513_183554.json
  python scripts/chunk_chat_topics.py "chat_*.json" --out chunks.json
  python scripts/chunk_chat_topics.py chat.json --gap-minutes 45 --depth-coef 0.7

Tuning:
  --gap-minutes  Larger  -> fewer, longer sessions.
  --block-size   Messages compared on each side of a candidate boundary.
  --depth-coef   Larger  -> more (finer) topic splits within a session.
  --smooth       Moving-average window over gap scores (1 = off).
  --min-chunk    Chunks smaller than this are merged into a neighbour.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from _rag import embed_texts, get_embedder, message_payload_text  # noqa: E402

USER_ID_RAW = os.environ.get("TELEGRAM_USER_ID")
TELEGRAM_USER_ID = int(USER_ID_RAW) if USER_ID_RAW else None


def is_user_message(msg: dict) -> bool | None:
    if TELEGRAM_USER_ID is None:
        return None
    return (msg.get("sender") or {}).get("id") == TELEGRAM_USER_ID


def sender_name(msg: dict) -> str:
    sender = msg.get("sender") or {}
    return sender.get("name") or sender.get("username") or str(sender.get("id", "?"))


def split_by_time_gap(messages: list[dict], gap_minutes: float) -> list[list[int]]:
    """Return sessions as lists of indices into `messages`, split on long silences."""
    if not messages:
        return []
    sessions: list[list[int]] = [[0]]
    prev = datetime.fromisoformat(messages[0]["date"])
    for i in range(1, len(messages)):
        cur = datetime.fromisoformat(messages[i]["date"])
        gap = (cur - prev).total_seconds() / 60
        if gap > gap_minutes:
            sessions.append([])
        sessions[-1].append(i)
        prev = cur
    return sessions


def block_mean(vectors: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of vectors[lo:hi], renormalised to unit length."""
    block = vectors[lo:hi]
    m = block.mean(axis=0)
    norm = np.linalg.norm(m)
    return m / norm if norm > 0 else m


def topic_boundaries(
    vectors: np.ndarray,
    block_size: int,
    depth_coef: float,
    smooth: int,
    sim_floor: float,
) -> list[int]:
    """
    TextTiling-style boundary detection over a sequence of unit embeddings.

    Returns indices `b` (1..n-1) meaning a new topic starts at position `b`.

    A gap only qualifies as a boundary if its cross-similarity is below
    `sim_floor` (the two sides are genuinely dissimilar) AND it is a deep,
    local valley relative to the rest of the session. The floor stops the
    relative-depth test from splitting where everything is on-topic — short
    chats rate ~0.92+ similar even across messages, so tiny local dips there
    are noise, not real topic changes.
    """
    n = len(vectors)
    if n < 2 * block_size:
        return []

    # gap_scores[i] = similarity across the boundary between item i and i+1.
    gap_scores = np.empty(n - 1)
    for i in range(n - 1):
        left = block_mean(vectors, max(0, i - block_size + 1), i + 1)
        right = block_mean(vectors, i + 1, min(n, i + 1 + block_size))
        gap_scores[i] = float(np.dot(left, right))

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        gap_scores = np.convolve(gap_scores, kernel, mode="same")

    # Depth score: how deep each valley is relative to its surrounding peaks.
    depths = np.zeros(n - 1)
    for i in range(n - 1):
        l = i
        while l > 0 and gap_scores[l - 1] > gap_scores[l]:
            l -= 1
        r = i
        while r < n - 2 and gap_scores[r + 1] > gap_scores[r]:
            r += 1
        depths[i] = (gap_scores[l] - gap_scores[i]) + (gap_scores[r] - gap_scores[i])

    cutoff = depths.mean() - depth_coef * depths.std()

    boundaries = []
    for i in range(n - 1):
        if gap_scores[i] >= sim_floor:
            continue
        if depths[i] <= cutoff:
            continue
        # Keep only genuine local minima of the similarity curve.
        left_ok = i == 0 or gap_scores[i] <= gap_scores[i - 1]
        right_ok = i == n - 2 or gap_scores[i] <= gap_scores[i + 1]
        if left_ok and right_ok:
            boundaries.append(i + 1)
    return boundaries


def segment_session(
    session_idx: list[int],
    messages: list[dict],
    embedder,
    block_size: int,
    depth_coef: float,
    smooth: int,
    sim_floor: float,
) -> list[list[int]]:
    """Split one session's message indices into topic-coherent runs."""
    # Only text-bearing messages drive the embedding segmentation.
    text_positions = [p for p, gi in enumerate(session_idx)
                      if message_payload_text(messages[gi])]
    if len(text_positions) < 2 * block_size:
        return [session_idx]

    texts = [message_payload_text(messages[session_idx[p]]) for p in text_positions]
    vectors = np.asarray(embed_texts(embedder, texts), dtype=np.float32)

    text_boundaries = topic_boundaries(vectors, block_size, depth_coef, smooth, sim_floor)

    # Map a boundary in text-message space to a position in the full session.
    cut_positions = sorted(text_positions[b] for b in text_boundaries)

    runs: list[list[int]] = []
    start = 0
    for cut in cut_positions:
        runs.append(session_idx[start:cut])
        start = cut
    runs.append(session_idx[start:])
    return [r for r in runs if r]


def merge_small(runs: list[list[int]], min_chunk: int) -> list[list[int]]:
    """Merge any run shorter than min_chunk into its previous (or next) neighbour."""
    if min_chunk <= 1:
        return runs
    merged: list[list[int]] = []
    for run in runs:
        if merged and len(run) < min_chunk:
            merged[-1].extend(run)
        else:
            merged.append(run)
    # First run may still be too small — fold it forward.
    if len(merged) > 1 and len(merged[0]) < min_chunk:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return merged


def build_chunk(chunk_id: int, session_id: int, idxs: list[int], messages: list[dict]) -> dict:
    msgs = [messages[i] for i in idxs]
    participants = sorted({sender_name(m) for m in msgs})
    out_messages = []
    for m in msgs:
        out_messages.append({
            "id": m.get("id"),
            "date": m.get("date"),
            "sender_id": (m.get("sender") or {}).get("id"),
            "sender_name": sender_name(m),
            "is_user": is_user_message(m),
            "text": message_payload_text(m),
        })
    return {
        "chunk_id": chunk_id,
        "session_id": session_id,
        "start_date": msgs[0].get("date"),
        "end_date": msgs[-1].get("date"),
        "message_count": len(msgs),
        "participants": participants,
        "messages": out_messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Chunk a parsed chat into topic groups")
    parser.add_argument("json_files", nargs="+", help="Parsed chat JSON (globs OK; all are merged)")
    parser.add_argument("--out", help="Output path (default: <input>.chunks.json)")
    parser.add_argument("--gap-minutes", type=float, default=30.0,
                        help="Silence longer than this starts a new session (default 30)")
    parser.add_argument("--block-size", type=int, default=4,
                        help="Messages compared on each side of a candidate boundary (default 4)")
    parser.add_argument("--depth-coef", type=float, default=0.5,
                        help="Higher -> more, finer topic splits (default 0.5)")
    parser.add_argument("--sim-floor", type=float, default=0.88,
                        help="Only split where cross-similarity is below this; "
                             "stops splitting on-topic chatter (default 0.88)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Moving-average window over gap scores; 1 = off (default 1)")
    parser.add_argument("--min-chunk", type=int, default=2,
                        help="Chunks smaller than this are merged into a neighbour (default 2)")
    args = parser.parse_args()

    files = []
    for pattern in args.json_files:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        print("No matching JSON files.", file=sys.stderr)
        sys.exit(1)
    if len(files) > 1 and not args.out:
        print("Multiple inputs need an explicit --out.", file=sys.stderr)
        sys.exit(1)

    # Load and merge messages (single chat assumed; chat meta from first file).
    chat_meta = None
    messages: list[dict] = []
    for file in files:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        if chat_meta is None:
            chat_meta = data.get("chat")
        messages.extend(data.get("messages", []))

    messages.sort(key=lambda m: m["date"])
    print(f"Loaded {len(messages)} messages from {len(files)} file(s).", file=sys.stderr)

    sessions = split_by_time_gap(messages, args.gap_minutes)
    print(f"Time-gap split -> {len(sessions)} sessions.", file=sys.stderr)

    embedder = get_embedder()

    chunks = []
    chunk_id = 0
    for session_id, session_idx in enumerate(sessions):
        runs = segment_session(
            session_idx, messages, embedder,
            args.block_size, args.depth_coef, args.smooth, args.sim_floor,
        )
        runs = merge_small(runs, args.min_chunk)
        for run in runs:
            chunks.append(build_chunk(chunk_id, session_id, run, messages))
            chunk_id += 1

    out_path = Path(args.out) if args.out else Path(
        str(Path(files[0]).with_suffix("")) + ".chunks.json")

    result = {
        "source": files,
        "chat": chat_meta,
        "generated_at": datetime.now().astimezone().isoformat(),
        "params": {
            "gap_minutes": args.gap_minutes,
            "block_size": args.block_size,
            "depth_coef": args.depth_coef,
            "sim_floor": args.sim_floor,
            "smooth": args.smooth,
            "min_chunk": args.min_chunk,
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
        },
        "total_messages": len(messages),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    sizes = [c["message_count"] for c in chunks]
    avg = sum(sizes) / len(sizes) if sizes else 0
    print(f"\nWrote {len(chunks)} chunks to {out_path}")
    print(f"Chunk size: min={min(sizes, default=0)} max={max(sizes, default=0)} avg={avg:.1f}")


if __name__ == "__main__":
    main()
