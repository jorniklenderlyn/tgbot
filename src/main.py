"""Entry point: starts the Aiogram control bot and the Telethon listener concurrently.

Ctrl-C handling:
  Both Telethon's `run_until_disconnected()` and aiogram's `start_polling()`
  block in their own internal loops and don't propagate KeyboardInterrupt
  cleanly. We install explicit signal handlers that cancel the orchestration
  task; they in turn cancel both child tasks and shut everything down.
"""

import asyncio
import signal
import sys

from openai import OpenAI

from src.agent import build_agent
from src.bot import ControlBot
from src.chat_prompts import load_chat_prompts
from src.config import ASSISTANT_LLM_MODEL, OPENROUTER_API_KEY
from src.drafts import DraftStore
from src.telegram_client import UserClient


async def amain(stop_event: asyncio.Event):
    print("Starting assistant...", file=sys.stderr)

    drafts = DraftStore()
    chat_prompts = load_chat_prompts()
    print(f"Chat prompts: {sorted(chat_prompts.keys()) or '(none)'}", file=sys.stderr)

    llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    agent = build_agent(chat_prompts=chat_prompts, llm=llm)

    user_client = UserClient(drafts, agent, on_draft=None, llm=llm, llm_model=ASSISTANT_LLM_MODEL)
    bot = ControlBot(drafts, send_reply_cb=user_client.send_reply)

    user_client.on_draft = bot.push_draft
    user_client.on_attention_alert = bot.push_attention_alert
    user_client.bind_manual_send(bot.delete_draft_message)

    await user_client.start()
    await bot.notify("🤖 Ассистент запущен. Драфты будут приходить сюда.")

    bot_task = asyncio.create_task(bot.run(), name="bot")
    tg_task = asyncio.create_task(user_client.run(), name="tg")
    stop_task = asyncio.create_task(stop_event.wait(), name="stop")

    done, pending = await asyncio.wait(
        {bot_task, tg_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Decide why we exited
    if stop_task in done:
        print("\n[main] shutdown signal received, stopping...", file=sys.stderr)
    else:
        # Either bot or telethon died on its own; surface the exception
        for t in done:
            exc = t.exception()
            if exc:
                print(f"[main] {t.get_name()} crashed: {exc}", file=sys.stderr)

    # Cancel everything that's still running
    for t in pending:
        t.cancel()

    # Best-effort graceful shutdown
    try:
        await bot.dp.stop_polling()
    except Exception:
        pass
    try:
        await user_client.tg.disconnect()
    except Exception:
        pass

    await asyncio.gather(*pending, return_exceptions=True)
    await bot.shutdown()
    print("[main] bye", file=sys.stderr)


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def _on_signal(sig_name: str):
        if not stop_event.is_set():
            print(f"\n[main] caught {sig_name}", file=sys.stderr)
            stop_event.set()
        else:
            # Second Ctrl-C: hard exit
            print("[main] second signal — forcing exit", file=sys.stderr)
            sys.exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
        except NotImplementedError:
            # Windows fallback
            signal.signal(sig, lambda s, f: _on_signal(signal.Signals(s).name))

    try:
        loop.run_until_complete(amain(stop_event))
    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt (fallback)", file=sys.stderr)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


if __name__ == "__main__":
    main()
