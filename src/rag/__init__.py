"""RAG infrastructure: embeddings, Qdrant access, and retrieval wrappers."""

from src.rag.rag import (  # noqa: F401
    embed_one,
    embed_texts,
    ensure_collection,
    get_embedder,
    get_qdrant,
    message_payload_text,
    render_message,
)
from src.rag.retrieval import (  # noqa: F401
    retrieve_facts,
    retrieve_style_examples,
)
