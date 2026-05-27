"""Shared style-profile helpers.

Turns a style-profile JSON (output of scripts/extract_style_profile.py) into a
compact persona description, and renders retrieved (incoming -> reply) pairs as
few-shot style examples. Used by both the CLI tools (scripts/style_chat.py,
eval) and the live assistant (src/agent/graph.py) so there is one source of
truth for how style is expressed to the LLM.
"""

import json
import os
import re

_MARKER_RE = re.compile(r"\[(?:изображение|видео|голос[^\]]*|медиа)[^\]]*\]")


def clean_text(text: str) -> str:
    """Strip media-caption markers ([изображение: ...]) left by the parser."""
    return _MARKER_RE.sub("", text or "").strip()


def render_persona(profile: dict) -> str:
    """Flatten the style-profile JSON into a compact persona description."""
    lines: list[str] = []
    s = profile.get("surface_style") or {}
    if s:
        lines.append("How they write (measured):")
        lines.append(f"- typical message: ~{s.get('avg_message_length_words')} words, "
                     f"density={s.get('message_density')}, fragmentation={s.get('fragmentation')} "
                     f"(often {s.get('messages_per_turn')} messages per turn)")
        lines.append(f"- capitalization: {s.get('capitalization')} "
                     f"(lowercase-start rate {s.get('lowercase_start_rate')})")
        lines.append(f"- terminal punctuation often omitted ({s.get('no_terminal_punctuation_rate')} of msgs)")
        if s.get("common_emojis"):
            lines.append(f"- emojis (rate {s.get('emoji_rate')}/msg): {' '.join(s['common_emojis'][:8])}")
        smile = s.get("happy_vs_sad_smileys") or {}
        if smile:
            lines.append(f"- bracket smileys: ) happy x{smile.get('happy')}, ( sad x{smile.get('sad')} "
                         f"(rate {s.get('bracket_smiley_rate')})")
        if s.get("slang"):
            lines.append(f"- slang/abbreviations: {', '.join(s['slang'][:15])}")
        if s.get("swear_roots"):
            lines.append(f"- swears (rate {s.get('swear_message_rate')}): {', '.join(s['swear_roots'])}")
        lines.append(f"- questions in {s.get('question_rate')} of msgs; "
                     f"ellipsis {'yes' if s.get('uses_ellipsis') else 'rare'}")

    lc = profile.get("linguistic_constructions") or {}
    if lc:
        lines.append("\nLinguistic habits:")
        if lc.get("frequent_openers"):
            lines.append(f"- openers: {', '.join(lc['frequent_openers'][:12])}")
        if lc.get("favorite_connectors"):
            lines.append(f"- connectors: {', '.join(lc['favorite_connectors'][:12])}")
        if lc.get("characteristic_phrases"):
            lines.append(f"- signature phrases: {', '.join(lc['characteristic_phrases'][:15])}")
        for key, label in [("reasoning_pattern", "reasoning"), ("question_style", "questions"),
                           ("explanation_style", "explaining"), ("correction_style", "corrections"),
                           ("rhythm", "rhythm")]:
            if lc.get(key):
                lines.append(f"- {label}: {lc[key]}")

    cb = profile.get("conversational_behavior") or {}
    if cb.get("summary"):
        lines.append(f"\nConversational behaviour: {cb['summary']}")
    for key, label in [("verbosity", "verbosity"), ("topic_switching", "topic-switching"),
                       ("follow_up_questions", "follow-ups")]:
        if cb.get(key):
            lines.append(f"- {label}: {cb[key]}")

    cs = profile.get("cognitive_style") or {}
    if cs.get("traits"):
        lines.append(f"\nCognitive traits: {', '.join(cs['traits'])}")
    if cs.get("summary"):
        lines.append(f"Cognitive style: {cs['summary']}")

    return "\n".join(lines)


def load_persona(path: str | None) -> str | None:
    """Load a style-profile JSON and render its persona. None if missing/empty."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    persona = render_persona(profile)
    return persona or None


def render_fewshot(examples) -> str:
    """Render retrieved style pairs (Qdrant ScoredPoints) into a few-shot block.

    Each example payload carries `incoming_text`, `reply_text` and (for the
    fact-redacted bank) a `situation` label. The text is already fact-free, so
    examples from other chats are safe to show; the situation lets the model
    judge whether an example fits the current case. Returns "" when empty.
    """
    rows = []
    for h in examples or []:
        payload = getattr(h, "payload", None) or (h if isinstance(h, dict) else {})
        inc = clean_text(payload.get("incoming_text", ""))
        rep = clean_text(payload.get("reply_text", ""))
        sit = (payload.get("situation") or "").strip()
        if inc and rep:
            rows.append((sit, inc, rep))
    if not rows:
        return ""
    lines = ["ПРИМЕРЫ ТВОЕГО СТИЛЯ В ПОХОЖИХ СИТУАЦИЯХ (факты убраны и заменены на "
             "[плейсхолдеры]; копируй ТОЛЬКО манеру, тон и длину, а содержание бери "
             "из текущего диалога):"]
    for sit, inc, rep in rows:
        prefix = f"[ситуация: {sit}] " if sit else ""
        lines.append(f"{prefix}Собеседник: {inc}\nТы: {rep}")
    return "\n\n".join(lines)
