"""Prompts used by the LangGraph agent — loaded from /prompts/*.md."""

from src.util import load_prompt

SYSTEM_PROMPT = load_prompt("system_assistant")
TRIAGE_PROMPT = load_prompt("triage")
CLASSIFY_PROMPT = load_prompt("classify_messages")
