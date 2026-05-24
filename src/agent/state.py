"""LangGraph state shared by all reply-generation nodes."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---- inputs ---- #
    chat_id: int
    chat_id_str: str
    chat_username: str | None
    sender_id: int
    sender_name: str
    incoming_text: str           # filled by transcribe_node (or copied from raw_text)
    raw_text: str
    media_kind: str | None       # "voice", "video_note", "video", "audio", None
    history_lines: list[str]

    # ---- retrieval ---- #
    style_examples: list[Any]    # qdrant ScoredPoint
    facts: list[Any]
    chat_prompt: str | None

    # ---- message classification ---- #
    message_type: str            # "story" | "questions" | "mixed" | "single"
    message_type_reason: str

    # ---- triage ---- #
    triage_classification: str   # "auto_reply" | "needs_attention"
    triage_reason: str

    # ---- working memory ---- #
    working_memory: str | None

    # ---- raw recent history ---- #
    raw_history: str | None

    # ---- output ---- #
    reply_text: str
    requires_approval: bool
    approval_reason: str
    error: str | None
