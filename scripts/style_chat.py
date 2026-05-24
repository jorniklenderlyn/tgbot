#!/usr/bin/env python3
"""
Chat as yourself: a style-imitating reply generator you can test in the console.

Pipeline (the pieces built earlier wired together):
  1. Load a style profile (output of extract_style_profile.py) -> persona.
  2. Each turn, embed the friend's incoming message and retrieve the most
     similar (incoming -> your-reply) pairs from the `tg_style` Qdrant
     collection (built by build_style_index.py) as live few-shot examples.
  3. Send {stable persona system prompt} + {conversation history} +
     {retrieved few-shot} + {incoming} to the LLM and print the reply.

You play the FRIEND (type their messages); the model replies as YOU, in your
style. This is the core of an AI assistant that texts like you.

Usage:
  python scripts/style_chat.py
  python scripts/style_chat.py --profile chat_ntsupkov_20260513_183554.style.json -k 6
  python scripts/style_chat.py --show-shots          # print retrieved examples each turn

Requires: Qdrant running with a populated `tg_style` collection, and
OPENROUTER_API_KEY set.
"""

import argparse
import glob
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.http import models as qm

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _rag import embed_one, get_embedder, get_qdrant  # noqa: E402
from src.prompt_loader import load_prompt  # noqa: E402

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
STYLE_COLLECTION = os.environ.get("STYLE_COLLECTION", "tg_style")
STYLE_LLM_MODEL = os.environ.get("STYLE_LLM_MODEL", "deepseek/deepseek-v4-flash")
STYLE_LLM_PROVIDER = os.environ.get("STYLE_LLM_PROVIDER", "DeepInfra")

MAX_HISTORY_TURNS = 12


def render_persona(profile: dict) -> str:
    """Flatten the style-profile JSON into a compact persona description."""
    lines = []
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


_MARKER_RE = __import__("re").compile(r"\[(?:изображение|видео|голос[^\]]*|медиа)[^\]]*\]")


def _clean(text: str) -> str:
    return _MARKER_RE.sub("", text or "").strip()


def retrieve_shots(qdrant, embedder, incoming: str, k: int) -> list[dict]:
    vec = embed_one(embedder, incoming, is_query=True)
    # Over-fetch, then drop pairs whose reply is just a media caption (no typed style).
    hits = qdrant.search(collection_name=STYLE_COLLECTION, query_vector=vec, limit=k * 3)
    shots = []
    for h in hits:
        p = h.payload
        reply = _clean(p.get("reply_text", ""))
        incoming_t = _clean(p.get("incoming_text", ""))
        if not reply or not incoming_t:
            continue
        shots.append({"incoming": incoming_t, "reply": reply, "score": h.score})
        if len(shots) >= k:
            break
    return shots


def shots_block(shots: list[dict]) -> str:
    parts = ["Похожие ситуации из прошлого — как ТЫ отвечал (ориентир по стилю, не по теме):"]
    for s in shots:
        parts.append(f"Собеседник: {s['incoming']}\nТы: {s['reply']}")
    return "\n\n".join(parts)


def find_default_profile() -> str | None:
    files = sorted(glob.glob("*.style.json"))
    return files[-1] if files else None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
        text = __import__("re").sub(r"^json\s*", "", text).strip()
    return text


def parse_bubbles(raw: str) -> list[str]:
    """Parse the model's reply into a list of message bubbles."""
    text = _strip_fences(raw)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            bubbles = [_clean(str(x)) for x in data]
            return [b for b in bubbles if b]
        if isinstance(data, str):
            return [_clean(data)] if _clean(data) else []
    except json.JSONDecodeError:
        pass
    # Fallback: treat non-empty lines as separate bubbles.
    bubbles = [_clean(ln) for ln in text.splitlines()]
    bubbles = [b for b in bubbles if b]
    return bubbles or [_clean(text)] if _clean(text) else []


def generate(client, model, extra, base_messages, temperature, hint: str | None) -> list[str]:
    msgs = list(base_messages)
    if hint:
        msgs.append({"role": "system",
                     "content": f"Прошлый ответ был отклонён. Исправь: {hint}"})
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature, extra_body=extra)
    return parse_bubbles(resp.choices[0].message.content.strip())


def critique(client, model, extra, incoming: str, history: list[dict], bubbles: list[str]) -> dict:
    """Return {'ok': bool, 'reason': str, 'suggestion': str}."""
    check_prompt = load_prompt("style_reply_check")
    ctx_lines = []
    for m in history[-6:]:
        tag = "Собеседник" if m["role"] == "user" else "Бот"
        ctx_lines.append(f"{tag}: {m['content']}")
    payload = (f"Контекст:\n{chr(10).join(ctx_lines) or '(нет)'}\n\n"
               f"Сообщение собеседника: {incoming}\n\n"
               f"Предложенный ответ бота (пузыри): {json.dumps(bubbles, ensure_ascii=False)}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": check_prompt},
                      {"role": "user", "content": payload}],
            temperature=0, extra_body=extra)
        verdict = json.loads(_strip_fences(resp.choices[0].message.content.strip()))
        return {"ok": bool(verdict.get("ok", True)),
                "reason": verdict.get("reason", ""),
                "suggestion": verdict.get("suggestion", "")}
    except Exception as e:
        # On checker failure, don't block the reply.
        return {"ok": True, "reason": f"(checker skipped: {e})", "suggestion": ""}


def main():
    parser = argparse.ArgumentParser(description="Console chat that replies in your style")
    parser.add_argument("--profile", help="Style profile JSON (default: latest *.style.json)")
    parser.add_argument("-k", "--top-k", type=int, default=6, help="Few-shot examples per turn (default 6)")
    parser.add_argument("--model", default=STYLE_LLM_MODEL, help=f"OpenRouter model (default {STYLE_LLM_MODEL})")
    parser.add_argument("--provider", default=STYLE_LLM_PROVIDER, help="Pin provider for cache stickiness")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default 0.8)")
    parser.add_argument("--show-shots", action="store_true", help="Print retrieved few-shot each turn")
    parser.add_argument("--no-check", action="store_true", help="Disable the appropriateness-check layer")
    parser.add_argument("--no-split", action="store_true", help="Join multi-bubble replies into one message")
    parser.add_argument("--max-retries", type=int, default=2, help="Regen attempts when the check fails (default 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show check verdicts and retries")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    profile_path = args.profile or find_default_profile()
    if not profile_path or not os.path.exists(profile_path):
        print("No style profile found. Run scripts/extract_style_profile.py first, "
              "or pass --profile.", file=sys.stderr)
        sys.exit(1)
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)

    persona = render_persona(profile)
    system_prompt = load_prompt("style_reply").replace("{{PROFILE}}", persona)

    embedder = get_embedder()
    qdrant = get_qdrant()
    try:
        info = qdrant.get_collection(STYLE_COLLECTION)
    except Exception as e:
        print(f"Collection '{STYLE_COLLECTION}' not found — run build_style_index.py. ({e})",
              file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    extra = {"usage": {"include": True}}
    if args.provider:
        extra["provider"] = {"order": [args.provider], "allow_fallbacks": False}

    who = (profile.get("sender") or {})
    name = who.get("name", "you") if isinstance(who, dict) else "you"
    print(f"Profile: {profile_path} ({name}) | shots from '{STYLE_COLLECTION}' "
          f"({info.points_count} pairs) | model {args.model}", file=sys.stderr)
    print("You are the FRIEND. Type a message; the assistant replies as "
          f"{name}. (Ctrl-C or 'exit' to quit, '/reset' to clear history.)\n", file=sys.stderr)

    history: list[dict] = []
    while True:
        try:
            incoming = input("друг> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not incoming:
            continue
        if incoming.lower() in ("exit", "quit"):
            break
        if incoming == "/reset":
            history.clear()
            print("(history cleared)\n", file=sys.stderr)
            continue

        shots = retrieve_shots(qdrant, embedder, incoming, args.top_k)
        if args.show_shots:
            print("\n--- few-shot ---", file=sys.stderr)
            for s in shots:
                print(f"  [{s['score']:.2f}] {s['incoming'][:40]!r} -> {s['reply'][:40]!r}",
                      file=sys.stderr)
            print("--- end ---\n", file=sys.stderr)

        base_messages = [{"role": "system", "content": system_prompt}]
        base_messages += history[-MAX_HISTORY_TURNS * 2:]
        if shots:
            base_messages.append({"role": "system", "content": shots_block(shots)})
        base_messages.append({"role": "user", "content": incoming})

        try:
            bubbles = generate(client, args.model, extra, base_messages, args.temperature, None)
            for attempt in range(args.max_retries if not args.no_check else 0):
                verdict = critique(client, args.model, extra, incoming, history, bubbles)
                if verdict["ok"]:
                    break
                if args.verbose:
                    print(f"  [check failed: {verdict['reason']} -> {verdict['suggestion']}]",
                          file=sys.stderr)
                bubbles = generate(client, args.model, extra, base_messages,
                                   args.temperature, verdict["suggestion"] or verdict["reason"])
        except Exception as e:
            print(f"[LLM error: {e}]\n", file=sys.stderr)
            continue

        if not bubbles:
            print("[empty reply]\n", file=sys.stderr)
            continue

        short = name.split()[0]
        if args.no_split:
            print(f"{short}> {' '.join(bubbles)}\n")
        else:
            for b in bubbles:
                print(f"{short}> {b}")
            print()

        history.append({"role": "user", "content": incoming})
        history.append({"role": "assistant", "content": "\n".join(bubbles)})


if __name__ == "__main__":
    main()
