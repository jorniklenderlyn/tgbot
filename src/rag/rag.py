"""Shared RAG helpers — rendering, embedding, Qdrant access."""

import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, QDRANT_URL


def render_message(msg: dict) -> str:
    """Render a parsed-chat JSON message into a single searchable text block."""
    sender = msg.get("sender", {})
    name = sender.get("name") or sender.get("username") or str(sender.get("id", "?"))
    date = msg.get("date", "")
    parts = [f"[{date}] {name}:"]

    if msg.get("forward"):
        fwd = msg["forward"]
        orig = fwd.get("original_sender_name") or fwd.get("original_sender_id") or "unknown"
        parts.append(f"(переслано от {orig})")

    if msg.get("is_reply"):
        parts.append(f"(ответ на сообщение #{msg.get('reply_to_msg_id')})")

    text = msg.get("text", "").strip()
    if text:
        parts.append(text)

    if msg.get("audio_transcript"):
        media_type = (msg.get("media") or {}).get("type")
        label = "видео" if media_type in ("video", "video_note") else "голосовое"
        parts.append(f"[{label}: {msg['audio_transcript']}]")

    if msg.get("image_description"):
        parts.append(f"[изображение: {msg['image_description']}]")
    elif msg.get("image_path"):
        parts.append("[изображение без описания]")

    media = msg.get("media")
    if media and media.get("type") not in ("voice_message", "audio", "photo", "video", "video_note"):
        parts.append(f"[медиа: {media.get('type')}]")

    return " ".join(parts)


def message_payload_text(msg: dict) -> str:
    """Just the textual payload — for style index where we don't want headers."""
    parts = []
    text = (msg.get("text") or "").strip()
    if text:
        parts.append(text)
    if msg.get("audio_transcript"):
        parts.append(msg["audio_transcript"])
    if msg.get("image_description"):
        parts.append(f"[изображение: {msg['image_description']}]")
    return " ".join(parts).strip()


def get_embedder(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or EMBEDDING_MODEL
    print(f"Loading embedding model '{name}'...", file=sys.stderr)
    return SentenceTransformer(name)


def embed_texts(model: SentenceTransformer, texts: list[str], is_query: bool = False) -> list[list[float]]:
    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_one(model: SentenceTransformer, text: str, is_query: bool = False) -> list[float]:
    return embed_texts(model, [text], is_query=is_query)[0]


def get_qdrant(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or QDRANT_URL)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    indexed_fields: list[str] | None = None,
):
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        return
    print(f"Creating collection '{collection_name}' (dim={vector_size})")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )
    for field in indexed_fields or []:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
