#!/usr/bin/env python3
"""
Build a writing-style profile for one chat participant.

Two layers:
  A. Surface style  — computed deterministically from raw typed text:
     casing, emoji, punctuation, message length, fragmentation/bursts,
     slang, abbreviations, swearing, elongation. (No API calls.)
  B/C/D. Linguistic constructions, conversational behaviour, cognitive style
     — interpretive, so delegated to an LLM (OpenRouter DeepSeek V4 Flash).
     The large instruction/schema is sent as a stable system prefix so
     DeepSeek's automatic prompt caching discounts repeat runs.

Input may be the RAW parsed chat dump (preferred — clean `text` field) or a
*.chunks.json from chunk_chat_topics.py. Raw is more accurate for surface
stats because it keeps typed text separate from voice transcripts / image
captions.

Usage:
  python scripts/extract_style_profile.py chat_ntsupkov_20260513_183554.json
  python scripts/extract_style_profile.py chat.json --sender me --out me_style.json
  python scripts/extract_style_profile.py chat.json --sender "Николай" --no-llm
  python scripts/extract_style_profile.py chat.json --sender all

--sender accepts: `me`, `all`, or any substring of a name/username/id.
`me` resolves via TELEGRAM_USER_ID if set, else the participant whose username
is not the chat title (2-party chats only).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.util.prompt_loader import load_prompt  # noqa: E402

USER_ID_RAW = os.environ.get("TELEGRAM_USER_ID")
TELEGRAM_USER_ID = int(USER_ID_RAW) if USER_ID_RAW else None
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
STYLE_LLM_MODEL = os.environ.get("STYLE_LLM_MODEL", "deepseek/deepseek-v4-flash")
# Pin one provider so DeepSeek's node-local prompt cache is reused on re-runs.
STYLE_LLM_PROVIDER = os.environ.get("STYLE_LLM_PROVIDER", "DeepInfra")

# Emoji ranges (pictographic). Excludes ASCII and invisible modifiers.
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "]",
    flags=re.UNICODE,
)

WORD_RE = re.compile(r"\w+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"[.!?…]+|\n+")
# A bracket "smiley" run, Russian style: )))  or  (((
BRACKET_SMILEY_RE = re.compile(r"(?<![\w(])([)(]{1,})(?![\w)])")

# High-precision slang / abbreviations (token must equal one of these).
RU_SLANG = {
    "кмк", "имхо", "лол", "кек", "рофл", "ржака", "бро", "ща", "щас", "че", "чё",
    "шо", "мб", "нзч", "плз", "пжлст", "спс", "омг", "изи", "кринж", "кринге",
    "збс", "хз", "гг", "ауф", "краш", "чел", "типа", "оки", "норм", "лан", "капец",
    "жесть", "хех", "мда", "пон", "ясн", "вродь", "нез", "ппц", "хосподи",
}
EN_SLANG = {
    "bro", "bruh", "nah", "fr", "ngl", "lol", "lmao", "lmfao", "tbh", "idk", "imo",
    "imho", "omg", "wtf", "yeah", "yep", "nope", "btw", "rn", "smth", "ikr",
}
# Profanity roots. Anchored at a word boundary (after optional common prefixes)
# so mid-word coincidences like "требую" -> "ебу" don't false-positive.
SWEAR_ROOTS = [
    "хуй", "хуе", "хуё", "хуя", "пизд", "ебл", "ёб", "еб", "бля", "блят", "блэт",
    "сук", "хер", "говн", "муда", "мраз", "гандон", "залуп", "дроч", "пидор",
    "пидар", "долбоёб", "fuck", "shit", "bitch",
]
SWEAR_RE = re.compile(
    r"\b(?:за|на|по|недо|пере|при|от|вы|до|у|в|об|разъ?|съ?)?(" +
    "|".join(SWEAR_ROOTS) + r")",
    re.IGNORECASE,
)


def load_input(path: str) -> tuple[list[dict], dict | None]:
    """Return (normalized_messages, chat_meta). Handles raw dumps and *.chunks.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chat_meta = data.get("chat")
    norm = []

    if "messages" in data:  # raw parsed dump
        for m in data["messages"]:
            sender = m.get("sender") or {}
            norm.append({
                "id": m.get("id"),
                "date": m.get("date"),
                "sender_id": sender.get("id"),
                "sender_name": sender.get("name") or sender.get("username") or str(sender.get("id")),
                "username": sender.get("username"),
                "text": (m.get("text") or "").strip(),  # typed text only
            })
    elif "chunks" in data:  # chunked file; text has merged transcript/caption markers
        marker = re.compile(r"\[(?:изображение|видео|голос[^\]]*|медиа)[^\]]*\]")
        for c in data["chunks"]:
            for m in c["messages"]:
                text = marker.sub("", m.get("text") or "").strip()
                norm.append({
                    "id": m.get("id"),
                    "date": m.get("date"),
                    "sender_id": m.get("sender_id"),
                    "sender_name": m.get("sender_name"),
                    "username": None,
                    "text": text,
                })
        print("Note: input is a chunks file; surface stats may include voice "
              "transcripts. Pass the raw dump for cleanest results.", file=sys.stderr)
    else:
        print("Unrecognised JSON (no 'messages' or 'chunks').", file=sys.stderr)
        sys.exit(1)

    norm.sort(key=lambda m: m["date"] or "")
    return norm, chat_meta


def list_senders(messages: list[dict]) -> list[tuple]:
    by_id = {}
    for m in messages:
        sid = m["sender_id"]
        if sid not in by_id:
            by_id[sid] = {"id": sid, "name": m["sender_name"],
                          "username": m.get("username"), "count": 0}
        by_id[sid]["count"] += 1
    return sorted(by_id.values(), key=lambda s: -s["count"])


def resolve_sender(messages: list[dict], chat_meta: dict | None, selector: str) -> set:
    """Return the set of sender_ids matching `selector`."""
    senders = list_senders(messages)
    all_ids = {s["id"] for s in senders}

    if selector == "all":
        return all_ids

    if selector == "me":
        if TELEGRAM_USER_ID is not None and TELEGRAM_USER_ID in all_ids:
            return {TELEGRAM_USER_ID}
        # Heuristic: in a 2-party chat, "me" is whoever's username != chat title.
        title = (chat_meta or {}).get("title", "")
        if len(senders) == 2 and title:
            others = [s for s in senders if (s.get("username") or "") != title
                      and title.lower() not in (s["name"] or "").lower()]
            if len(others) == 1:
                return {others[0]["id"]}
        print("Cannot resolve 'me' (set TELEGRAM_USER_ID or pass --sender NAME).\n"
              "Participants:", file=sys.stderr)
        for s in senders:
            print(f"  {s['count']:5d}  {s['name']}  @{s.get('username')}  id={s['id']}",
                  file=sys.stderr)
        sys.exit(1)

    # Substring match on name / username / id.
    sel = selector.lower()
    matched = {s["id"] for s in senders
               if sel in (s["name"] or "").lower()
               or sel in (s.get("username") or "").lower()
               or sel == str(s["id"])}
    if not matched:
        print(f"No sender matches '{selector}'.", file=sys.stderr)
        sys.exit(1)
    return matched


# ---------------------------------------------------------------------------
# A. Surface style (deterministic)
# ---------------------------------------------------------------------------

def count_emojis(text: str) -> list[str]:
    return EMOJI_RE.findall(text)


def surface_style(texts: list[str], full_stream: list[dict], target_ids: set) -> dict:
    texts = [t for t in texts if t]
    n = len(texts)
    if n == 0:
        return {"message_count": 0}

    char_lens, word_lens, sent_lens = [], [], []
    emoji_counter = Counter()
    emoji_total = 0
    ellipsis_msgs = 0
    lower_start = 0
    allcaps_words = 0
    total_words = 0
    multi_excl = multi_q = q_msgs = excl_msgs = 0
    bracket_smiley_msgs = 0
    happy_smiley = sad_smiley = 0
    elongation_words = 0
    punct_chars = 0
    total_chars = 0
    no_terminal_punct = 0
    slang_counter = Counter()
    abbr_counter = Counter()
    swear_msgs = 0
    swear_roots_found = Counter()

    punct_set = set(".,!?;:—-…()\"'«»")

    for t in texts:
        char_lens.append(len(t))
        total_chars += len(t)
        words = WORD_RE.findall(t)
        word_lens.append(len(words))
        total_words += len(words)

        ems = count_emojis(t)
        emoji_total += len(ems)
        emoji_counter.update(ems)

        if "..." in t or "…" in t:
            ellipsis_msgs += 1

        first_alpha = next((c for c in t if c.isalpha()), None)
        if first_alpha and first_alpha.islower():
            lower_start += 1

        if t.rstrip() and t.rstrip()[-1] not in ".!?…)(":
            no_terminal_punct += 1

        if "!!" in t:
            multi_excl += 1
        if "??" in t:
            multi_q += 1
        if "?" in t:
            q_msgs += 1
        if "!" in t:
            excl_msgs += 1

        for run in BRACKET_SMILEY_RE.findall(t):
            if ")" in run:
                happy_smiley += 1
            if "(" in run:
                sad_smiley += 1
        if BRACKET_SMILEY_RE.search(t):
            bracket_smiley_msgs += 1

        for w in words:
            if len(w) >= 2 and w.isupper() and not w.isdigit():
                allcaps_words += 1
            if re.search(r"(.)\1\1", w):  # 3+ repeated char (даааа, нееет)
                elongation_words += 1
            lw = w.lower()
            if lw in RU_SLANG or lw in EN_SLANG:
                (slang_counter if lw in RU_SLANG else abbr_counter)[lw] += 1

        roots_here = [m.group(1).lower() for m in SWEAR_RE.finditer(t)]
        if roots_here:
            swear_msgs += 1
            swear_roots_found.update(roots_here)

        for c in t:
            if c in punct_set:
                punct_chars += 1

        for s in SENT_SPLIT_RE.split(t):
            sw = WORD_RE.findall(s)
            if sw:
                sent_lens.append(len(sw))

    # Bursts / fragmentation across the ordered stream (target sender only).
    runs = []
    cur = 0
    for m in full_stream:
        if m["sender_id"] in target_ids and m["text"]:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    avg_burst = round(sum(runs) / len(runs), 2) if runs else 0
    multi_msg_turns = sum(1 for r in runs if r > 1)
    burst_share = round(sum(r for r in runs if r > 1) / sum(runs), 3) if runs else 0

    avg_words = sum(word_lens) / n
    lower_start_rate = round(lower_start / n, 3)

    def density(aw):
        if aw <= 4:
            return "compact"
        if aw <= 12:
            return "moderate"
        return "verbose"

    if lower_start_rate >= 0.7:
        cap = "minimal"
    elif lower_start_rate <= 0.2:
        cap = "standard"
    else:
        cap = "mixed"

    return {
        "message_count": n,
        "avg_message_length_chars": round(sum(char_lens) / n, 1),
        "avg_message_length_words": round(avg_words, 2),
        "median_message_length_words": median(word_lens),
        "p90_message_length_words": sorted(word_lens)[int(0.9 * (n - 1))],
        "avg_sentence_length_words": round(sum(sent_lens) / len(sent_lens), 2) if sent_lens else 0,
        "message_density": density(avg_words),
        "lowercase": lower_start_rate >= 0.6,
        "capitalization": cap,
        "lowercase_start_rate": lower_start_rate,
        "allcaps_word_rate": round(allcaps_words / max(total_words, 1), 4),
        "emoji_rate": round(emoji_total / n, 3),
        "common_emojis": [e for e, _ in emoji_counter.most_common(10)],
        "emoji_counts": dict(emoji_counter.most_common(10)),
        "uses_ellipsis": ellipsis_msgs / n >= 0.05,
        "ellipsis_rate": round(ellipsis_msgs / n, 3),
        "bracket_smiley_rate": round(bracket_smiley_msgs / n, 3),
        "happy_vs_sad_smileys": {"happy": happy_smiley, "sad": sad_smiley},
        "question_rate": round(q_msgs / n, 3),
        "exclamation_rate": round(excl_msgs / n, 3),
        "multi_exclamation_rate": round(multi_excl / n, 3),
        "multi_question_rate": round(multi_q / n, 3),
        "elongation_rate": round(elongation_words / max(total_words, 1), 4),
        "punctuation_density": round(punct_chars / max(total_chars, 1), 3),
        "no_terminal_punctuation_rate": round(no_terminal_punct / n, 3),
        "slang": [w for w, _ in slang_counter.most_common(20)],
        "slang_counts": dict(slang_counter.most_common(20)),
        "abbreviations": [w for w, _ in abbr_counter.most_common(20)],
        "internet_slang": [w for w, _ in (slang_counter + abbr_counter).most_common(20)],
        "swear_message_rate": round(swear_msgs / n, 3),
        "swear_roots": [r for r, _ in swear_roots_found.most_common(15)],
        "messages_per_turn": avg_burst,
        "multi_message_turns": multi_msg_turns,
        "burst_share": burst_share,
        "fragmentation": "high" if avg_burst >= 2 or avg_words <= 4 else "moderate" if avg_words <= 9 else "low",
    }


# ---------------------------------------------------------------------------
# B/C/D. LLM profile
# ---------------------------------------------------------------------------

def build_transcript(messages: list[dict], target_ids: set, max_chars: int) -> str:
    lines = []
    for m in messages:
        if not m["text"]:
            continue
        tag = "ME" if m["sender_id"] in target_ids else "THEM"
        lines.append(f"{tag}: {m['text']}")

    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full

    # Spread the budget across the whole timeline: keep every k-th line.
    k = max(2, len(full) // max_chars + 1)
    sampled = lines[::k]
    out = "\n".join(sampled)
    while len(out) > max_chars and sampled:
        sampled = sampled[:-1]
        out = "\n".join(sampled)
    return out


def llm_profile(transcript: str, model: str, provider: str | None) -> dict:
    from openai import OpenAI

    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set — skipping LLM layer (use --no-llm to silence).",
              file=sys.stderr)
        return {}

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    system_prompt = load_prompt("extract_style_profile")

    extra = {"usage": {"include": True}}
    # The prompt cache is node-local; OpenRouter load-balances across providers
    # by default, so repeat runs land on different nodes and never reuse the
    # cache. Hard-pinning one provider (no fallbacks) makes routing sticky, so
    # re-running the same profile reuses ~95% of the prompt at a steep discount.
    if provider:
        extra["provider"] = {"order": [provider], "allow_fallbacks": False}

    print(f"Calling {model} ({len(transcript)} chars of transcript)...", file=sys.stderr)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            # Stable prefix sent first so the provider can cache it across runs.
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        extra_body=extra,
    )

    usage = getattr(resp, "usage", None)
    if usage:
        cached = getattr(usage, "prompt_tokens_details", None)
        cached_n = getattr(cached, "cached_tokens", None) if cached else None
        served_by = getattr(resp, "provider", None)
        print(f"Tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
              + (f" cached={cached_n}" if cached_n is not None else "")
              + (f" via {served_by}" if served_by else ""), file=sys.stderr)

    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        content = re.sub(r"^json\s*", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        print("LLM did not return valid JSON; storing raw text.", file=sys.stderr)
        return {"_raw": content}


def main():
    parser = argparse.ArgumentParser(description="Extract a writing-style profile for one participant")
    parser.add_argument("input", help="Raw parsed chat JSON (preferred) or a *.chunks.json")
    parser.add_argument("--sender", default="me",
                        help="'me' (default), 'all', or a name/username/id substring")
    parser.add_argument("--out", help="Output JSON path (default: <input>.style.json)")
    parser.add_argument("--no-llm", action="store_true", help="Surface stats only, skip the LLM layer")
    parser.add_argument("--model", default=STYLE_LLM_MODEL, help=f"OpenRouter model (default {STYLE_LLM_MODEL})")
    parser.add_argument("--provider", default=STYLE_LLM_PROVIDER,
                        help=f"Pin OpenRouter provider for cache stickiness (default {STYLE_LLM_PROVIDER}; "
                             "pass '' to let OpenRouter load-balance)")
    parser.add_argument("--max-chars", type=int, default=60000,
                        help="Transcript budget sent to the LLM (default 60000)")
    args = parser.parse_args()

    messages, chat_meta = load_input(args.input)
    target_ids = resolve_sender(messages, chat_meta, args.sender)

    senders = {s["id"]: s for s in list_senders(messages)}
    target_info = [
        {"id": sid, "name": senders[sid]["name"], "username": senders[sid].get("username"),
         "message_count": senders[sid]["count"]}
        for sid in target_ids
    ]
    label = ", ".join(t["name"] for t in target_info)
    print(f"Profiling: {label} ({sum(t['message_count'] for t in target_info)} messages)",
          file=sys.stderr)

    target_texts = [m["text"] for m in messages if m["sender_id"] in target_ids]
    surface = surface_style(target_texts, messages, target_ids)

    profile = {}
    if not args.no_llm:
        transcript = build_transcript(messages, target_ids, args.max_chars)
        profile = llm_profile(transcript, args.model, args.provider or None)

    result = {
        "source": args.input,
        "chat": chat_meta,
        "sender": target_info if len(target_info) > 1 else (target_info[0] if target_info else None),
        "sender_selector": args.sender,
        "llm_model": None if args.no_llm else args.model,
        "surface_style": surface,
        "linguistic_constructions": profile.get("linguistic_constructions"),
        "conversational_behavior": profile.get("conversational_behavior"),
        "cognitive_style": profile.get("cognitive_style"),
        "evidence": profile.get("evidence"),
    }
    if profile.get("_raw"):
        result["llm_raw"] = profile["_raw"]

    out_path = Path(args.out) if args.out else Path(
        str(Path(args.input).with_suffix("")) + ".style.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote style profile to {out_path}")


if __name__ == "__main__":
    main()
