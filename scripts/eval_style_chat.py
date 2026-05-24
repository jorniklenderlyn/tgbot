#!/usr/bin/env python3
"""
LLM-as-judge evaluation for the style-chat pipeline.

Runs a set of incoming messages through the same generate (+ optional check)
pipeline as style_chat.py, then an LLM judge scores each reply on style
fidelity, appropriateness, naturalness, and burst quality (1-5). Prints an
aggregate report so you can A/B test knobs (--temperature, -k, --no-check).

Test set, in priority order:
  --from-dump chat.json   sample real (friend -> your-reply) pairs; the real
                          reply is shown to the judge as a reference.
  --tests FILE            one incoming message per line.
  (default)               a built-in curated set covering common categories.

Usage:
  python scripts/eval_style_chat.py
  python scripts/eval_style_chat.py --from-dump chat_ntsupkov_20260513_183554.json -n 25
  python scripts/eval_style_chat.py --no-check --temperature 1.0   # compare configs
"""

import argparse
import glob
import json
import os
import random
import sys
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import style_chat as sc  # noqa: E402  (sibling script; reuse its helpers)
from _rag import get_embedder, get_qdrant  # noqa: E402
from src.prompt_loader import load_prompt  # noqa: E402

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DEFAULT_TESTS = [
    ("greeting", "Привет"),
    ("greeting", "здарова бро"),
    ("question_plan", "во сколько завтра встречаемся?"),
    ("question_fact", "ты лабу по матану сделал?"),
    ("request", "скинь конспект по дискретке"),
    ("request_help", "помоги разобраться с питоном, не работает код"),
    ("banter", "ну ты и лошара))"),
    ("banter", "споришь на сотку что не сдашь?"),
    ("emotional_bad", "блин у меня всё плохо, завалил экзамен"),
    ("emotional_good", "я сдал на отл!!"),
    ("plans", "го вечером в качалку"),
    ("open", "как тебе универ?"),
    ("open", "что нового"),
    ("topic_shift", "кстати ты смотрел новый фильм?"),
    ("smalltalk", "что делаешь"),
]


def build_pipeline(args):
    profile_path = args.profile or (sorted(glob.glob("*.style.json")) or [None])[-1]
    if not profile_path or not os.path.exists(profile_path):
        print("No style profile found (run extract_style_profile.py or pass --profile).",
              file=sys.stderr)
        sys.exit(1)
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    persona = sc.render_persona(profile)
    system_prompt = load_prompt("style_reply").replace("{{PROFILE}}", persona)
    embedder = get_embedder()
    qdrant = get_qdrant()
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    extra = {"usage": {"include": True}}
    if args.provider:
        extra["provider"] = {"order": [args.provider], "allow_fallbacks": False}
    return profile_path, profile, persona, system_prompt, embedder, qdrant, client, extra


def make_reply(args, system_prompt, embedder, qdrant, client, extra, incoming):
    shots = sc.retrieve_shots(qdrant, embedder, incoming, args.top_k)
    base = [{"role": "system", "content": system_prompt}]
    if shots:
        base.append({"role": "system", "content": sc.shots_block(shots)})
    base.append({"role": "user", "content": incoming})
    bubbles = sc.generate(client, args.model, extra, base, args.temperature, None)
    if not args.no_check:
        for _ in range(args.max_retries):
            v = sc.critique(client, args.model, extra, incoming, [], bubbles)
            if v["ok"]:
                break
            bubbles = sc.generate(client, args.model, extra, base, args.temperature,
                                  v["suggestion"] or v["reason"])
    return bubbles


def judge(client, judge_model, extra, persona, incoming, bubbles, reference):
    judge_prompt = load_prompt("style_judge")
    payload = (f"=== PROFILE ===\n{persona}\n\n"
               f"Friend's message: {incoming}\n"
               f"Bot reply (bubbles): {json.dumps(bubbles, ensure_ascii=False)}")
    if reference:
        payload += f"\nReal reply (reference): {reference}"
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "system", "content": judge_prompt},
                  {"role": "user", "content": payload}],
        temperature=0, extra_body=extra)
    raw = sc._strip_fences(resp.choices[0].message.content.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"overall": None, "rationale": f"(judge parse fail: {raw[:80]})"}


def sample_from_dump(path, n):
    """Real (friend -> your-reply) pairs as test items with a reference reply."""
    import build_style_index as bsi  # imports require TELEGRAM_USER_ID (set in .env)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = bsi.find_pairs(data["messages"])
    random.shuffle(pairs)
    items = []
    for p in pairs[:n]:
        items.append(("real", p["incoming_text"], p["reply_text"]))
    return items


CRITERIA = ["style_fidelity", "appropriateness", "naturalness", "bursts", "overall"]


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge eval for the style-chat pipeline")
    parser.add_argument("--profile", help="Style profile JSON (default: latest *.style.json)")
    parser.add_argument("-k", "--top-k", type=int, default=6)
    parser.add_argument("--model", default=sc.STYLE_LLM_MODEL, help="Generator model")
    parser.add_argument("--judge-model", default=sc.STYLE_LLM_MODEL,
                        help="Judge model (default = generator; a different model reduces self-bias)")
    parser.add_argument("--provider", default=sc.STYLE_LLM_PROVIDER)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-check", action="store_true", help="Eval without the appropriateness layer")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--from-dump", help="Sample real friend->you pairs from this chat JSON")
    parser.add_argument("-n", type=int, default=20, help="How many pairs to sample with --from-dump")
    parser.add_argument("--tests", help="File with one incoming message per line")
    parser.add_argument("--out", default="style_eval.json", help="Report output path")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    _, _, persona, system_prompt, embedder, qdrant, client, extra = build_pipeline(args)

    if args.from_dump:
        items = sample_from_dump(args.from_dump, args.n)
    elif args.tests:
        with open(args.tests, encoding="utf-8") as f:
            items = [("custom", ln.strip(), None) for ln in f if ln.strip()]
    else:
        items = [(cat, txt, None) for cat, txt in DEFAULT_TESTS]

    print(f"Evaluating {len(items)} items | gen={args.model} judge={args.judge_model} "
          f"| check={'off' if args.no_check else 'on'} temp={args.temperature} k={args.top_k}",
          file=sys.stderr)

    results = []
    for i, (cat, incoming, reference) in enumerate(items, 1):
        bubbles = make_reply(args, system_prompt, embedder, qdrant, client, extra, incoming)
        verdict = judge(client, args.judge_model, extra, persona, incoming, bubbles, reference)
        results.append({"category": cat, "incoming": incoming, "reference": reference,
                        "reply": bubbles, "scores": verdict})
        ov = verdict.get("overall")
        print(f"[{i:2d}/{len(items)}] {cat:14} overall={ov} | "
              f"{incoming[:30]!r} -> {bubbles}", file=sys.stderr)

    # Aggregate.
    def collect(key):
        vals = [r["scores"].get(key) for r in results if isinstance(r["scores"].get(key), (int, float))]
        return round(mean(vals), 2) if vals else None

    agg = {c: collect(c) for c in CRITERIA}
    by_cat = {}
    for r in results:
        ov = r["scores"].get("overall")
        if isinstance(ov, (int, float)):
            by_cat.setdefault(r["category"], []).append(ov)
    cat_means = {c: round(mean(v), 2) for c, v in sorted(by_cat.items())}
    worst = sorted([r for r in results if isinstance(r["scores"].get("overall"), (int, float))],
                   key=lambda r: r["scores"]["overall"])[:3]

    report = {"config": {"model": args.model, "judge_model": args.judge_model,
                         "check": not args.no_check, "temperature": args.temperature,
                         "top_k": args.top_k, "n_items": len(items)},
              "aggregate": agg, "by_category": cat_means, "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== AGGREGATE (1-5) ===")
    for c in CRITERIA:
        print(f"  {c:16} {agg[c]}")
    print("\n=== BY CATEGORY (overall) ===")
    for c, m in cat_means.items():
        print(f"  {c:16} {m}")
    print("\n=== WORST 3 ===")
    for r in worst:
        print(f"  [{r['scores']['overall']}] {r['incoming'][:35]!r} -> {r['reply']}")
        print(f"       {r['scores'].get('rationale','')}")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
