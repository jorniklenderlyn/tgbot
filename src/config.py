"""Central configuration loaded from .env.

Required fields are validated **lazily** via `require()` — that way RAG-only
tools (parse_chat, ingest, query) can import shared modules without supplying
the assistant-only credentials (BOT_TOKEN, TELEGRAM_USER_ID).
"""

import os

from dotenv import load_dotenv

load_dotenv()


def require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required env var: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val


def require_int(name: str) -> int:
    return int(require(name))


# ------------- Telegram (user account) — accessed by assistant.py -------- #
def telegram_api_id() -> int:        return require_int("TELEGRAM_API_ID")
def telegram_api_hash() -> str:      return require("TELEGRAM_API_HASH")
def telegram_phone() -> str:         return require("TELEGRAM_PHONE")
def telegram_user_id() -> int:       return require_int("TELEGRAM_USER_ID")

# ------------- Aiogram (control bot) ------------------------------------- #
def bot_token() -> str:              return require("BOT_TOKEN")
def bot_owner_id() -> int:
    raw = os.environ.get("BOT_OWNER_ID")
    return int(raw) if raw else telegram_user_id()


def bot_user_id() -> int:
    """The control bot's own Telegram user ID, derived from BOT_TOKEN."""
    return int(bot_token().split(":", 1)[0])

# ------------- Plain settings (have defaults, safe to read at import) ---- #
SESSION_NAME = os.environ.get("TELEGRAM_SESSION", "session")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
ASSISTANT_LLM_MODEL = (
    os.environ.get("ASSISTANT_LLM_MODEL")
    or OPENROUTER_LLM_MODEL
)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "tg_chat")
STYLE_COLLECTION = os.environ.get("STYLE_COLLECTION", "tg_style")

WHITELIST_FILE = os.environ.get("WHITELIST_FILE", "whitelist.json")
CHAT_PROMPTS_DIR = os.environ.get("CHAT_PROMPTS_DIR", "chat_prompts")
HISTORY_DEPTH = int(os.environ.get("ASSISTANT_HISTORY_DEPTH", "10"))
TRANSCRIBE_METHOD = os.environ.get("ASSISTANT_TRANSCRIBE", "local")
DRAFT_TTL_SECONDS = int(os.environ.get("DRAFT_TTL_SECONDS", "3600"))

WORKING_MEMORY_DIR = os.environ.get("WORKING_MEMORY_DIR", ".chat_contexts")
WORKING_MEMORY_MAX_TOKENS = int(os.environ.get("WORKING_MEMORY_MAX_TOKENS", "4000"))
MESSAGE_DEBOUNCE_SECONDS = float(os.environ.get("MESSAGE_DEBOUNCE_SECONDS", "6"))

# Smart debounce: extend wait when messages arrive in quick bursts
DEBOUNCE_FAST_GAP = float(os.environ.get("DEBOUNCE_FAST_GAP", "2.0"))
DEBOUNCE_EXTENDED_SECONDS = float(os.environ.get("DEBOUNCE_EXTENDED_SECONDS", "8.0"))
DEBOUNCE_HARD_CAP_SECONDS = float(os.environ.get("DEBOUNCE_HARD_CAP_SECONDS", "30.0"))

# Raw recent messages passed as plain context to the LLM (both sides)
RAW_HISTORY_DEPTH = int(os.environ.get("RAW_HISTORY_DEPTH", "10"))

# Human-like reply delay range (seconds), randomised within this range
REPLY_DELAY_MIN = float(os.environ.get("REPLY_DELAY_MIN", "1.0"))
REPLY_DELAY_MAX = float(os.environ.get("REPLY_DELAY_MAX", "3.0"))
