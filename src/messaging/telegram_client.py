"""Telethon user-account client.

Responsibilities:
  - Listen for incoming PRIVATE messages, transcribe voice/video, debounce
    consecutive messages from the same chat, invoke the LangGraph agent on
    the combined text, push the resulting draft to the Aiogram control bot.
  - Listen for OUTGOING messages in private chats — if the user replies
    manually, delete the pending bot draft and cancel any pending debounced
    agent run so we don't accidentally double-send.
  - Expose `send_reply(chat_id, text)` for the control bot to use when
    forwarding an approved draft.
"""

import asyncio
import random
import sys
import time

from openai import OpenAI
from telethon import TelegramClient, events, functions
from telethon.tl.types import (
    DocumentAttributeAudio,
    DocumentAttributeVideo,
    MessageMediaDocument,
)

from src.agent import run_agent
from src.config import (
    ASSISTANT_LLM_MODEL,
    DEBOUNCE_EXTENDED_SECONDS,
    DEBOUNCE_FAST_GAP,
    DEBOUNCE_HARD_CAP_SECONDS,
    MESSAGE_DEBOUNCE_SECONDS,
    RAW_HISTORY_DEPTH,
    REPLY_DELAY_MAX,
    REPLY_DELAY_MIN,
    SESSION_NAME,
    TRANSCRIBE_METHOD,
    bot_user_id,
    telegram_api_hash,
    telegram_api_id,
    telegram_phone,
)
from src.memory import DraftStore, resolve_mode
from src.messaging.transcription import transcribe_video_message, transcribe_voice_message
from src.memory import (
    build_initial_memory,
    load_memory,
    save_memory,
    update_memory,
)


def _media_kind(message) -> str | None:
    media = getattr(message, "media", None)
    if not isinstance(media, MessageMediaDocument) or media.document is None:
        return None
    for attr in media.document.attributes:
        if isinstance(attr, DocumentAttributeAudio):
            return "voice" if attr.voice else "audio"
        if isinstance(attr, DocumentAttributeVideo):
            return "video_note" if attr.round_message else "video"
    return None


async def _render_incoming(message) -> tuple[str, str | None]:
    """Returns (incoming_text, media_kind)."""
    text = (message.message or "").strip()
    kind = _media_kind(message)
    if kind in ("voice", "audio"):
        t = await transcribe_voice_message(message.client, message, TRANSCRIBE_METHOD)
        if t:
            text = (text + " " if text else "") + f"[голосовое: {t}]"
    elif kind in ("video", "video_note"):
        t = await transcribe_video_message(message.client, message, TRANSCRIBE_METHOD)
        if t:
            text = (text + " " if text else "") + f"[видео: {t}]"
    return text, kind


class UserClient:
    def __init__(self, drafts: DraftStore, agent, on_draft, llm: OpenAI | None = None, llm_model: str | None = None):
        """
        on_draft: async callable (Draft) -> int|None — pushes draft to bot,
                  returns the bot message_id.
        """
        self.tg = TelegramClient(SESSION_NAME, telegram_api_id(), telegram_api_hash())
        self.drafts = drafts
        self.agent = agent
        self.on_draft = on_draft
        self.on_attention_alert = None  # async (chat_id, username, sender, text, reason) -> None
        self._on_manual_send = None
        self._control_bot_id = bot_user_id()
        self._llm = llm
        self._llm_model = llm_model or ASSISTANT_LLM_MODEL
        self._my_id: int | None = None
        # Per-chat debounce buffers. Each entry:
        #   {"texts": [str, ...], "task": Task, "last_event": Message,
        #    "username": str|None, "sender_name": str, "sender_id": int,
        #    "mode": str, "last_kind": str|None,
        #    "first_ts": float, "last_ts": float, "msg_timestamps": [float]}
        self._pending: dict[int, dict] = {}

    def bind_manual_send(self, cb):
        """cb: async (chat_id:int, bot_message_id:int) → None. Called when user types manually."""
        self._on_manual_send = cb

    async def start(self):
        await self.tg.start(phone=telegram_phone())
        me = await self.tg.get_me()
        self._my_id = me.id
        print(f"[tg] logged in as @{me.username} (id={me.id})", file=sys.stderr)
        self._register_handlers()

    async def run(self):
        print("[tg] listening for messages...", file=sys.stderr)
        await self.tg.run_until_disconnected()

    async def send_reply(self, chat_id: int, text: str) -> bool:
        try:
            await self.tg.send_message(chat_id, text)
            return True
        except Exception as e:
            print(f"[tg] send_reply failed for {chat_id}: {e}", file=sys.stderr)
            return False

    # --------------------------------------------------------------------- #

    def _cancel_pending(self, chat_id: int) -> None:
        entry = self._pending.pop(chat_id, None)
        if entry and entry.get("task"):
            entry["task"].cancel()

    def _compute_wait(self, entry: dict) -> float:
        """Adaptive debounce: shorter if slow typing, longer if burst, capped."""
        timestamps = entry["msg_timestamps"]
        elapsed = time.monotonic() - entry["first_ts"]
        remaining_cap = max(0.5, DEBOUNCE_HARD_CAP_SECONDS - elapsed)

        if len(timestamps) >= 2:
            last_gap = timestamps[-1] - timestamps[-2]
            if last_gap < DEBOUNCE_FAST_GAP:
                return min(DEBOUNCE_EXTENDED_SECONDS, remaining_cap)

        return min(MESSAGE_DEBOUNCE_SECONDS, remaining_cap)

    async def _debounce_then_process(self, chat_id: int) -> None:
        try:
            entry = self._pending.get(chat_id)
            if not entry:
                return
            wait = self._compute_wait(entry)
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            return

        entry = self._pending.pop(chat_id, None)
        if entry is None:
            return

        try:
            await self._process_chat(chat_id, entry)
        except Exception as e:
            print(f"[tg] _process_chat error: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)

    async def _fetch_raw_history(self, chat_id: int) -> str | None:
        """Fetch last N messages from both sides as plain text context."""
        try:
            lines: list[str] = []
            async for msg in self.tg.iter_messages(chat_id, limit=RAW_HISTORY_DEPTH):
                text = (msg.message or "").strip()
                if not text:
                    continue
                prefix = "Я" if msg.sender_id == self._my_id else "Собеседник"
                lines.append(f"{prefix}: {text}")
            if not lines:
                return None
            lines.reverse()
            return "\n".join(lines)
        except Exception as e:
            print(f"[tg] raw history fetch failed for {chat_id}: {e}", file=sys.stderr)
            return None

    async def _send_typing(self, chat_id: int) -> None:
        try:
            from telethon.tl.types import SendMessageTypingAction
            await self.tg(functions.messages.SetTypingRequest(
                peer=chat_id,
                action=SendMessageTypingAction(),
            ))
        except Exception as e:
            print(f"[tg] typing indicator failed: {e}", file=sys.stderr)

    async def _keep_typing(self, chat_id: int, stop_event: asyncio.Event) -> None:
        """Continuously send typing indicators every 5s until stop_event is set."""
        while not stop_event.is_set():
            await self._send_typing(chat_id)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    async def _process_chat(self, chat_id: int, entry: dict) -> None:
        combined = "\n".join(entry["texts"]).strip()
        if not combined:
            return

        username = entry["username"]
        sender_name = entry["sender_name"]
        sender_id = entry["sender_id"]
        mode = entry["mode"]
        kind = entry["last_kind"]

        print(f"[tg] processing {len(entry['texts'])} msg(s) from @{username or chat_id}: {combined[:120]}",
              file=sys.stderr)

        # Start typing indicator
        typing_stop = asyncio.Event()
        typing_task = asyncio.create_task(self._keep_typing(chat_id, typing_stop))

        # Bootstrap working memory for new chats
        if self._llm and load_memory(chat_id) is None:
            print(f"[memory] building initial memory for {chat_id}...", file=sys.stderr)
            mem = await build_initial_memory(
                self.tg, chat_id, self._my_id, self._llm, self._llm_model,
            )
            if mem:
                save_memory(chat_id, mem)
                print(f"[memory] initial memory saved for {chat_id}", file=sys.stderr)

        # Fetch raw recent messages for context
        raw_history = await self._fetch_raw_history(chat_id)

        state = {
            "chat_id": chat_id,
            "chat_id_str": str(chat_id),
            "chat_username": username,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "incoming_text": combined,
            "raw_text": combined,
            "media_kind": kind,
            "raw_history": raw_history,
        }

        result = await run_agent(self.agent, state)

        # Stop typing indicator
        typing_stop.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass
        triage = result.get("triage_classification", "needs_attention")
        is_attention = triage == "needs_attention"
        reply = (result.get("reply_text") or "").strip()

        if not reply:
            if is_attention:
                reason = result.get("triage_reason") or "требует внимания"
                print(f"[tg] needs_attention (no reply): {reason}", file=sys.stderr)
                if self.on_attention_alert:
                    await self.on_attention_alert(
                        chat_id, username, sender_name, combined, reason)
                if self._llm:
                    exchange = f"Собеседник: {combined}"
                    asyncio.create_task(
                        update_memory(chat_id, None, exchange, self._llm, self._llm_model))
            else:
                print("[tg] empty reply, skipping", file=sys.stderr)
            return

        if is_attention:
            approval_reason = result.get("triage_reason") or "требует внимания"
        else:
            approval_reason = result.get("approval_reason") or ""

        if mode == "autonomous" and not is_attention and not result.get("requires_approval"):
            delay = random.uniform(REPLY_DELAY_MIN, REPLY_DELAY_MAX)
            await asyncio.sleep(delay)
            await self.send_reply(chat_id, reply)
            print(f"[tg] auto-sent to {username or chat_id} (delay={delay:.1f}s)", file=sys.stderr)
        else:
            draft = self.drafts.create(
                chat_id=chat_id,
                chat_username=username,
                sender_name=sender_name,
                incoming_text=combined,
                reply_text=reply,
                approval_reason=approval_reason,
                is_attention=is_attention,
            )
            await self.on_draft(draft)
            print(f"[tg] draft {draft.id} → bot (attention={is_attention})", file=sys.stderr)

        if self._llm:
            exchange = f"Собеседник: {combined}\nЯ: {reply}"
            asyncio.create_task(
                update_memory(chat_id, None, exchange, self._llm, self._llm_model))

    def _register_handlers(self):
        @self.tg.on(events.NewMessage(incoming=True))
        async def on_incoming(event):
            try:
                self.drafts.purge_expired()

                if not event.is_private:
                    return

                chat = await event.get_chat()
                sender = await event.get_sender()
                chat_id = event.chat_id

                if chat_id == self._control_bot_id or sender.id == self._control_bot_id:
                    return
                if getattr(sender, "bot", False):
                    return

                username = getattr(chat, "username", None) or getattr(sender, "username", None)
                sender_name = (
                    f"{(sender.first_name or '').strip()} {(sender.last_name or '').strip()}".strip()
                    or (sender.username or str(sender.id))
                )

                mode = resolve_mode(chat_id)
                if mode is None:
                    return

                incoming_text, kind = await _render_incoming(event.message)
                if not incoming_text:
                    return

                print(f"[tg] in @{username or chat_id} ({sender_name}) [{mode}]: {incoming_text[:120]}",
                      file=sys.stderr)

                # Mark as read immediately so the sender sees the "read" indicator.
                try:
                    await self.tg.send_read_acknowledge(chat_id, event.message)
                except Exception as e:
                    print(f"[tg] mark-read failed: {e}", file=sys.stderr)

                # --- debounce: append to buffer, restart timer --- #
                existing = self._pending.get(chat_id)
                if existing and existing.get("task"):
                    existing["task"].cancel()

                now = time.monotonic()
                if existing:
                    existing["texts"].append(incoming_text)
                    existing["last_event"] = event
                    existing["last_kind"] = kind
                    existing["username"] = username
                    existing["sender_name"] = sender_name
                    existing["sender_id"] = sender.id
                    existing["mode"] = mode
                    existing["last_ts"] = now
                    existing["msg_timestamps"].append(now)
                    entry = existing
                else:
                    entry = {
                        "texts": [incoming_text],
                        "last_event": event,
                        "username": username,
                        "sender_name": sender_name,
                        "sender_id": sender.id,
                        "mode": mode,
                        "last_kind": kind,
                        "task": None,
                        "first_ts": now,
                        "last_ts": now,
                        "msg_timestamps": [now],
                    }
                    self._pending[chat_id] = entry

                entry["task"] = asyncio.create_task(self._debounce_then_process(chat_id))

            except Exception as e:
                print(f"[tg] on_incoming error: {e}", file=sys.stderr)
                import traceback; traceback.print_exc(file=sys.stderr)

        @self.tg.on(events.NewMessage(outgoing=True))
        async def on_outgoing(event):
            try:
                if not event.is_private:
                    return
                chat_id = event.chat_id

                # User typed manually — cancel any pending debounced agent run.
                self._cancel_pending(chat_id)

                pending = self.drafts.pop_for_chat(chat_id)
                if pending and self._on_manual_send and pending.bot_message_id:
                    await self._on_manual_send(pending.bot_message_id)
                    print(f"[tg] manual reply in {chat_id} → cancelled draft {pending.id}",
                          file=sys.stderr)

                text = (event.message.message or "").strip()
                if text and self._llm:
                    exchange = f"Я: {text}"
                    asyncio.create_task(
                        update_memory(chat_id, None, exchange, self._llm, self._llm_model))
            except Exception as e:
                print(f"[tg] on_outgoing error: {e}", file=sys.stderr)
