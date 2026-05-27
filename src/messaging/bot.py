"""Aiogram control bot.

Shows drafts with inline buttons (✅ Approve / ✏️ Edit / 🗑 Skip).
Sending approved replies is delegated to the Telethon client through the
`send_reply` async callback passed at startup.
"""

import asyncio
import sys
from typing import Awaitable, Callable

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.config import bot_owner_id, bot_token
from src.memory import Draft, DraftStore


SendReplyCb = Callable[[int, str], Awaitable[bool]]


# Per-user state for the "edit" flow: { user_id → draft_id awaiting edit }
_edit_pending: dict[int, str] = {}


def _approve_kb(draft_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Отправить", callback_data=f"ok:{draft_id}"),
        InlineKeyboardButton(text="✏️ Изменить", callback_data=f"edit:{draft_id}"),
        InlineKeyboardButton(text="🗑 Пропустить", callback_data=f"skip:{draft_id}"),
    ]])


def _format_draft(d: Draft) -> str:
    handle = f"@{d.chat_username}" if d.chat_username else f"id={d.chat_id}"
    header = "🔔 <b>Требует внимания</b>\n\n" if d.is_attention else ""
    reason = f"\n\n⚠️ <i>{_escape(d.approval_reason)}</i>" if d.approval_reason else ""
    return (
        f"{header}"
        f"💬 <b>{d.sender_name}</b> ({handle})\n"
        f"<blockquote>{_escape(d.incoming_text)}</blockquote>\n"
        f"→ <code>{d.id}</code>\n"
        f"<b>{_escape(d.reply_text)}</b>"
        f"{reason}"
    )


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class ControlBot:
    def __init__(self, drafts: DraftStore, send_reply_cb: SendReplyCb):
        from aiogram.client.default import DefaultBotProperties
        self._owner_id = bot_owner_id()
        self.bot = Bot(token=bot_token(),
                       default=DefaultBotProperties(parse_mode="HTML"))
        self.dp = Dispatcher()
        self.drafts = drafts
        self.send_reply = send_reply_cb
        self._register()

    # ----- public API used by Telethon listener ----- #
    async def push_draft(self, draft: Draft) -> int | None:
        try:
            msg = await self.bot.send_message(
                self._owner_id,
                _format_draft(draft),
                reply_markup=_approve_kb(draft.id),
            )
            self.drafts.attach_bot_message(draft.id, msg.message_id)
            return msg.message_id
        except Exception as e:
            print(f"[bot] push_draft failed: {e}", file=sys.stderr)
            return None

    async def delete_draft_message(self, bot_message_id: int):
        try:
            await self.bot.delete_message(chat_id=self._owner_id, message_id=bot_message_id)
        except Exception as e:
            # message might already be gone; not fatal
            print(f"[bot] delete_message failed for {bot_message_id}: {e}", file=sys.stderr)

    async def push_attention_alert(self, chat_id: int, chat_username: str | None,
                                     sender_name: str, incoming_text: str, reason: str):
        handle = f"@{chat_username}" if chat_username else f"id={chat_id}"
        text = (
            f"🔔 <b>Требует внимания</b>\n\n"
            f"<b>{_escape(sender_name)}</b> ({handle})\n"
            f"<blockquote>{_escape(incoming_text)}</blockquote>\n"
            f"⚠️ <i>{_escape(reason)}</i>"
        )
        try:
            msg = await self.bot.send_message(
                self._owner_id, text,
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="👁 Прочитано", callback_data=f"ack:{chat_id}"),
                ]]),
            )
            return msg.message_id
        except Exception as e:
            print(f"[bot] push_attention_alert failed: {e}", file=sys.stderr)
            return None

    async def notify(self, text: str):
        try:
            await self.bot.send_message(self._owner_id, text)
        except Exception as e:
            print(f"[bot] notify failed: {e}", file=sys.stderr)

    async def run(self):
        print(f"[bot] polling started (owner={self._owner_id})", file=sys.stderr)
        await self.dp.start_polling(self.bot)

    async def shutdown(self):
        await self.bot.session.close()

    # ----- internal handlers ----- #
    def _register(self):
        dp = self.dp

        @dp.message(Command("start"))
        async def cmd_start(m: Message):
            if m.from_user.id != self._owner_id:
                return
            await m.answer("Привет. Я буду присылать сюда драфты ответов. "
                           "Жми кнопки или просто ответь сам — драфт удалится.")

        @dp.message(Command("ping"))
        async def cmd_ping(m: Message):
            if m.from_user.id != self._owner_id:
                return
            await m.answer("pong")

        @dp.callback_query(F.data.startswith("ok:"))
        async def on_ok(c: CallbackQuery):
            if c.from_user.id != self._owner_id:
                await c.answer("not for you", show_alert=False)
                return
            draft_id = c.data.split(":", 1)[1]
            draft = self.drafts.pop(draft_id)
            if not draft:
                await c.answer("draft не найден / истёк", show_alert=True)
                if c.message:
                    await c.message.edit_reply_markup(reply_markup=None)
                return
            ok = await self.send_reply(draft.chat_id, draft.reply_text)
            await c.answer("отправлено" if ok else "ошибка отправки", show_alert=not ok)
            if c.message:
                marker = "✅ отправлено" if ok else "❌ ошибка"
                await c.message.edit_text(f"{_format_draft(draft)}\n\n<i>{marker}</i>",
                                          reply_markup=None)

        @dp.callback_query(F.data.startswith("skip:"))
        async def on_skip(c: CallbackQuery):
            if c.from_user.id != self._owner_id:
                await c.answer("not for you")
                return
            draft_id = c.data.split(":", 1)[1]
            draft = self.drafts.pop(draft_id)
            await c.answer("пропущено")
            if c.message:
                if draft:
                    await c.message.edit_text(f"{_format_draft(draft)}\n\n<i>🗑 пропущено</i>",
                                              reply_markup=None)
                else:
                    await c.message.edit_reply_markup(reply_markup=None)

        @dp.callback_query(F.data.startswith("ack:"))
        async def on_ack(c: CallbackQuery):
            if c.from_user.id != self._owner_id:
                await c.answer("not for you")
                return
            await c.answer("отмечено")
            if c.message:
                await c.message.edit_reply_markup(reply_markup=None)

        @dp.callback_query(F.data.startswith("edit:"))
        async def on_edit(c: CallbackQuery):
            if c.from_user.id != self._owner_id:
                await c.answer("not for you")
                return
            draft_id = c.data.split(":", 1)[1]
            draft = self.drafts.get(draft_id)
            if not draft:
                await c.answer("draft не найден", show_alert=True)
                return
            _edit_pending[c.from_user.id] = draft_id
            await c.answer("Жду новый текст ответом сюда. /cancel чтобы отменить.")

        @dp.message(Command("cancel"))
        async def cmd_cancel(m: Message):
            if m.from_user.id != self._owner_id:
                return
            if _edit_pending.pop(m.from_user.id, None):
                await m.answer("Редактирование отменено.")
            else:
                await m.answer("Нечего отменять.")

        @dp.message(F.text & ~F.text.startswith("/"))
        async def on_text(m: Message):
            if m.from_user.id != self._owner_id:
                return
            draft_id = _edit_pending.pop(m.from_user.id, None)
            if not draft_id:
                return  # not in edit mode — ignore unrelated chatter
            draft = self.drafts.get(draft_id)
            if not draft:
                await m.answer("draft не найден / истёк")
                return

            new_text = m.text.strip()
            self.drafts.update_reply(draft_id, new_text)

            # rewrite the draft card in place with the new text + KEEP buttons
            if draft.bot_message_id:
                try:
                    await self.bot.edit_message_text(
                        chat_id=self._owner_id,
                        message_id=draft.bot_message_id,
                        text=f"{_format_draft(draft)}\n\n<i>✏️ отредактировано — можно отправлять</i>",
                        reply_markup=_approve_kb(draft.id),
                    )
                except Exception as e:
                    print(f"[bot] edit failed: {e}", file=sys.stderr)

            # tidy: remove the user's typed-edit message
            try:
                await m.delete()
            except Exception:
                pass
