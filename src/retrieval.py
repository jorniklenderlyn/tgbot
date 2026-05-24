"""Qdrant retrieval wrappers used by the agent."""

import sys

from qdrant_client.http import models as qm

from src.config import QDRANT_COLLECTION, STYLE_COLLECTION


def retrieve_style_examples(qdrant, vector, chat_id: str, k: int = 5):
    flt = qm.Filter(must=[qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id))])
    try:
        return qdrant.search(
            collection_name=STYLE_COLLECTION,
            query_vector=vector,
            limit=k,
            query_filter=flt,
        )
    except Exception as e:
        print(f"[warn] style search failed: {e}", file=sys.stderr)
        return []


def retrieve_facts(qdrant, vector, chat_id: str, k: int = 3):
    flt = qm.Filter(must=[
        qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
        qm.FieldCondition(key="type", match=qm.MatchValue(value="fact")),
    ])
    try:
        return qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=k,
            query_filter=flt,
        )
    except Exception as e:
        print(f"[warn] facts search failed: {e}", file=sys.stderr)
        return []
