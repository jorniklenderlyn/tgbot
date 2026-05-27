# Style Pipeline — Extraction & Application

How the assistant learns to text like you and then uses that in live replies.

There are two halves:

- **Extraction** (offline): turn a Telegram chat dump into a reusable *style
  profile* and a *few-shot index*.
- **Application** (online): at reply time, combine the profile (persona) with
  retrieved few-shot examples and generate a message in your style.

```
                          EXTRACTION (offline)                         APPLICATION (online)
┌────────────────────┐
│ chat_*.json (dump) │
└─────────┬──────────┘
          │
          ├──▶ scripts/chunk_chat_topics.py ──▶ *.chunks.json   (topic segments, for analysis)
          │
          ├──▶ scripts/extract_style_profile.py ──▶ *.style.json  ─────────┐  PERSONA (global "how I write")
          │                                                                 │
          └──▶ scripts/build_style_index.py ──▶ Qdrant `tg_style` ──────┐   │  FEW-SHOT (per-chat examples)
                                                                        │   │
                                                                        ▼   ▼
                                              src/agent/graph.py  _generate_node
                                              (persona + few-shot + context) ─▶ reply
                                                         ▲
                                   incoming Telegram message (via src/messaging/telegram_client.py)
```

---

## 1. Extraction

### 1a. Topic chunking — `scripts/chunk_chat_topics.py`
Optional, used for analysis. Splits the dump into topic-coherent chunks:

1. **Time-gap split** — a silence longer than `--gap-minutes` (default 30) starts
   a new session.
2. **Embedding split** — within a session, a TextTiling-style pass over local
   `multilingual-e5` embeddings finds semantic dips and cuts there. A
   `--sim-floor` (default 0.88) blocks splits where both sides are still very
   similar (short chats rate ~0.92+ similar, so tiny dips are noise, not topic
   changes).

```
.venv/bin/python scripts/chunk_chat_topics.py chat_*.json
```
Output: `*.chunks.json` (chunk metadata + messages). Not required by the
assistant — it's for inspecting the conversation, not for generation.

### 1b. Style profile — `scripts/extract_style_profile.py`
Produces `*.style.json`, the **persona**. Two layers:

- **A. Surface style (deterministic, no API).** Regex/counting over your raw
  typed `text` (kept separate from voice transcripts / image captions for
  accuracy): casing, emoji rate + top emojis, bracket-smileys (`)` vs `(`),
  message length, ellipsis, question/exclamation rates, elongation (`дааа`),
  slang/abbreviations, swearing (anchored regex to avoid false positives like
  `требую`→`ебу`), and **fragmentation/bursts** (messages per turn).
- **B/C/D. Interpretive (LLM).** Linguistic constructions (openers, connectors,
  reasoning/question/correction style, rhythm), conversational behaviour, and
  cognitive style — grounded in verbatim quotes. Uses `deepseek/deepseek-v4-flash`
  via OpenRouter. Prompt: `prompts/extract_style_profile.md`.

Whose style: `--sender me` (default) resolves via `TELEGRAM_USER_ID`, else a
2-party heuristic (the participant whose username ≠ chat title). Also `all` or a
name/username/id substring.

```
.venv/bin/python scripts/extract_style_profile.py chat_*.json --sender me
.venv/bin/python scripts/extract_style_profile.py chat_*.json --no-llm   # surface only
```

**Caching:** the static instruction/schema is sent as a stable system prefix.
DeepSeek's prompt cache is node-local, so OpenRouter is hard-pinned to one
provider (`STYLE_LLM_PROVIDER`, default `DeepInfra`) — otherwise load-balancing
across nodes defeats the cache. Re-running the same profile then reuses ~95%+ of
the prompt at a steep discount (verified ~99% hit).

### 1c. Few-shot index — `scripts/build_style_index.py`
Builds the Qdrant `tg_style` collection: a reusable **style bank** of
*(incoming → your reply)* pairs. Requires `TELEGRAM_USER_ID`.

By default each pair goes through an LLM **redaction + context** pass
(`prompts/redact_style_pair.md`, batched + cached, provider-pinned):

- **Fact redaction** — specific facts are replaced by placeholders
  (`[имя]`, `[место]`, `[время]`, `[тема]`, `[ссылка]`, `[объект]`) while the
  *style* is preserved exactly (casing, punctuation, emoji, smileys, slang,
  length). The raw fact-bearing text is **not stored**. This is what makes an
  example safe to reuse in a *different* chat without leaking private data.
- **Situation label** — a short, fact-free `situation` describing the context
  and intent (e.g. "ответ на вопрос о времени встречи", "реакция на хорошую
  новость"). Stored alongside the pair so callers can judge whether an example
  fits the current case.

The embedding is computed on the **redacted incoming** text (so a raw incoming
message at query time still matches). `--no-redact` stores raw text (legacy).

```
.venv/bin/python scripts/build_style_index.py chat_*.json --reset
```
At query time we embed the incoming message and find the most similar past
(redacted) incomings, then surface the (redacted) replies + their situation —
those become live few-shot examples. Because content is fact-free, retrieval can
safely span **all chats**, not just the current one.

---

## 2. Application

### 2a. Shared rendering — `src/util/style.py`
One source of truth for how style is expressed to the LLM:

- `render_persona(profile)` → compact persona text from `*.style.json`.
- `load_persona(path)` → load + render, or `None` if missing.
- `render_fewshot(examples)` → format retrieved `(incoming, reply)` pairs into a
  Russian few-shot block, dropping media-only replies.

Used by both the CLI tools and the live assistant, so they never diverge.

### 2b. Live assistant — `src/agent/graph.py`
The LangGraph flow:
```
load_chat_prompt → load_working_memory → classify_messages → triage → generate → END
```
Style is injected in **`_generate_node`**. The message stack sent to the LLM:

1. `SYSTEM_PROMPT` (`prompts/system_assistant.md`) — "you are the user, reply in
   your style ... per the examples below".
2. **Persona** — `ТВОЙ СТИЛЬ ПИСЬМА (профиль)` (global, from `*.style.json`).
3. **Few-shot** — `ПРИМЕРЫ ТВОЕГО СТИЛЯ В ПОХОЖИХ СИТУАЦИЯХ`, retrieved per
   message from `tg_style`. Content is fact-redacted and each example carries a
   `[ситуация: …]` label, so retrieval spans all chats (`STYLE_CROSS_CHAT=1`)
   and the model copies only manner/tone, taking content from the live dialog.
   Set `STYLE_CROSS_CHAT=0` to restrict examples to the current chat.
4. Per-chat prompt rules, working memory, recent raw history.
5. The incoming message.

Generation stays JSON: `{reply, requires_approval, approval_reason}`. **Triage**
(risk keywords + LLM) still gates anything about plans/money/dates to
`needs_attention` so it requires your approval in the control bot.

Wiring:
- `build_agent(..., qdrant, embedder, persona, fewshot_k)` — style is active when
  these are supplied; otherwise plain mode.
- `src/main.py::_init_style()` loads the persona (`STYLE_PROFILE_FILE` or latest
  `*.style.json`), Qdrant and the embedder at startup. Any failure (no profile,
  Qdrant down, `tg_style` missing) **degrades gracefully** to persona-only or
  plain mode — the assistant always starts.

Per-chat few-shot vs global persona: the persona captures your overall voice;
the few-shot adapts to how you specifically talk to each person.

### 2c. CLI tools (for testing the application)
- **`scripts/style_chat.py`** — chat in the console; you type as the friend, it
  replies as you. Adds two layers on top of retrieval: **multi-message bursts**
  (JSON array of 1–4 bubbles) and an **appropriateness check** (a critic call
  that regenerates degenerate replies, e.g. a bare `)` to a greeting). These live
  in the CLI tool, not yet in the live agent (see Limitations).
  ```
  .venv/bin/python scripts/style_chat.py -v --show-shots
  ```
- **`scripts/eval_style_chat.py`** — LLM-as-judge harness. Runs a test set
  through the pipeline and scores style fidelity, appropriateness, naturalness
  and bursts (1–5). Test set = built-in curated, `--tests FILE`, or `--from-dump`
  (real friend→you pairs with your real reply as reference).
  ```
  .venv/bin/python scripts/eval_style_chat.py --judge-model google/gemini-2.5-flash
  ```
  Note: a same-model judge is lenient (self-bias) — use a different judge family
  for trustworthy numbers.

---

## 3. Configuration (`.env`)

| Var | Default | Purpose |
|-----|---------|---------|
| `TELEGRAM_USER_ID` | — | Which messages are yours (extraction + index). |
| `STYLE_LLM_MODEL` | `deepseek/deepseek-v4-flash` | LLM for profile B/C/D and CLI chat. |
| `STYLE_LLM_PROVIDER` | `DeepInfra` | Hard-pinned provider for prompt-cache stickiness. Empty = load-balance (no cache). |
| `STYLE_ENABLED` | `1` | `0` = assistant runs without persona/few-shot. |
| `STYLE_PROFILE_FILE` | _(auto)_ | Persona profile path; empty picks latest `*.style.json`. |
| `STYLE_FEWSHOT_K` | `6` | Few-shot examples retrieved per message. |
| `STYLE_CROSS_CHAT` | `1` | Retrieve style examples across all chats (safe — bank is fact-redacted). `0` = current chat only. |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Local embedder (e5 query/passage prefixes). |
| `STYLE_COLLECTION` | `tg_style` | Qdrant collection of few-shot pairs. |

---

## 4. End-to-end runbook

```
# 0. Qdrant up
docker compose up -d qdrant

# 1. (optional) inspect topics
.venv/bin/python scripts/chunk_chat_topics.py chat_ntsupkov_*.json

# 2. build the persona profile
.venv/bin/python scripts/extract_style_profile.py chat_ntsupkov_*.json --sender me

# 3. build the few-shot index
.venv/bin/python scripts/build_style_index.py chat_ntsupkov_*.json

# 4. test in the console
.venv/bin/python scripts/style_chat.py -v

# 5. score the pipeline
.venv/bin/python scripts/eval_style_chat.py --judge-model google/gemini-2.5-flash

# 6. run the live assistant (loads persona + few-shot automatically)
.venv/bin/python -m src.main
```

---

## 5. Limitations & next steps

- The live agent (`_generate_node`) does **not** yet run the CLI's
  appropriateness-check or multi-bubble splitting — it emits one JSON reply. Both
  could be ported in (extra LLM call for the check; split on output) if desired.
- Few-shot quality depends on retrieval similarity. For incoming topics absent
  from your history, examples weaken and the model leans on the persona profile.
- The persona is derived from one chat; it's a good global proxy but not
  per-relationship. Per-chat tone is covered by the chat-filtered few-shot.
- Embeddings here are e5 (asymmetric: `query:` for the incoming message,
  `passage:` for indexed messages). Keep that consistent if you swap models.
