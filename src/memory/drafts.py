"""In-memory store of pending drafts shared between Telethon listener and Aiogram bot."""

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from threading import RLock

from src.config import DRAFT_TTL_SECONDS, HISTORY_DEPTH


@dataclass
class Draft:
    id: str
    chat_id: int                # Telethon chat id (private chat == user id)
    chat_username: str | None
    sender_name: str
    incoming_text: str
    reply_text: str
    approval_reason: str
    created_at: float
    is_attention: bool = False
    bot_message_id: int | None = None    # id of the message in the bot chat (so we can edit/delete)


class DraftStore:
    """Thread-safe-ish container for pending drafts + per-chat short message history."""

    def __init__(self):
        self._drafts: dict[str, Draft] = {}
        self._by_chat: dict[int, str] = {}   # chat_id → draft_id (latest pending)
        self._history: dict[int, deque] = {}
        self._lock = RLock()

    # ----- drafts ----- #
    def create(self, *, chat_id: int, chat_username: str | None, sender_name: str,
               incoming_text: str, reply_text: str, approval_reason: str,
               is_attention: bool = False) -> Draft:
        with self._lock:
            existing_id = self._by_chat.get(chat_id)
            if existing_id:
                # supersede the previous pending draft for this chat
                self._drafts.pop(existing_id, None)
            d = Draft(
                id=uuid.uuid4().hex[:6],
                chat_id=chat_id,
                chat_username=chat_username,
                sender_name=sender_name,
                incoming_text=incoming_text,
                reply_text=reply_text,
                approval_reason=approval_reason,
                created_at=time.time(),
                is_attention=is_attention,
            )
            self._drafts[d.id] = d
            self._by_chat[chat_id] = d.id
            return d

    def get(self, draft_id: str) -> Draft | None:
        with self._lock:
            return self._drafts.get(draft_id)

    def get_by_chat(self, chat_id: int) -> Draft | None:
        with self._lock:
            did = self._by_chat.get(chat_id)
            return self._drafts.get(did) if did else None

    def attach_bot_message(self, draft_id: str, bot_message_id: int):
        with self._lock:
            d = self._drafts.get(draft_id)
            if d:
                d.bot_message_id = bot_message_id

    def update_reply(self, draft_id: str, new_text: str):
        with self._lock:
            d = self._drafts.get(draft_id)
            if d:
                d.reply_text = new_text

    def pop(self, draft_id: str) -> Draft | None:
        with self._lock:
            d = self._drafts.pop(draft_id, None)
            if d and self._by_chat.get(d.chat_id) == draft_id:
                self._by_chat.pop(d.chat_id, None)
            return d

    def pop_for_chat(self, chat_id: int) -> Draft | None:
        with self._lock:
            did = self._by_chat.pop(chat_id, None)
            if not did:
                return None
            return self._drafts.pop(did, None)

    def purge_expired(self) -> list[Draft]:
        cutoff = time.time() - DRAFT_TTL_SECONDS
        expired = []
        with self._lock:
            for did, d in list(self._drafts.items()):
                if d.created_at < cutoff:
                    self._drafts.pop(did, None)
                    if self._by_chat.get(d.chat_id) == did:
                        self._by_chat.pop(d.chat_id, None)
                    expired.append(d)
        return expired

    # ----- short history per chat (used as live context) ----- #
    def remember(self, chat_id: int, line: str):
        with self._lock:
            dq = self._history.setdefault(chat_id, deque(maxlen=HISTORY_DEPTH * 3))
            dq.append(line)

    def history(self, chat_id: int) -> list[str]:
        with self._lock:
            dq = self._history.get(chat_id)
            return list(dq)[-HISTORY_DEPTH:] if dq else []
