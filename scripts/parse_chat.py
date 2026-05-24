#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaDocument,
    MessageMediaPhoto,
    MessageMediaGeo,
    MessageMediaContact,
    MessageMediaPoll,
    MessageMediaWebPage,
    DocumentAttributeAudio,
    DocumentAttributeFilename,
    DocumentAttributeVideo,
    DocumentAttributeSticker,
    PeerUser,
    PeerChat,
    PeerChannel,
)

load_dotenv()

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
PHONE = os.environ["TELEGRAM_PHONE"]

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def parse_date(s: str) -> datetime:
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid date format: {s}. Use DD.MM.YYYY or YYYY-MM-DD")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse Telegram chat messages to JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--last",
        type=str,
        help="Parse messages from last N time units, e.g. 1d, 12h, 7d, 2w",
    )
    group.add_argument(
        "--from-date",
        type=parse_date,
        help="Start date (DD.MM.YYYY or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to-date",
        type=parse_date,
        help="End date (DD.MM.YYYY or YYYY-MM-DD). Defaults to now.",
    )
    parser.add_argument(
        "chat",
        type=str,
        help="Chat to parse: username, invite link, or numeric ID",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path. Defaults to chat_<name>_<timestamp>.json",
    )
    parser.add_argument(
        "--transcribe",
        type=str,
        choices=["openrouter", "local", "none"],
        default="none",
        help="Audio transcription method: openrouter (API), local (faster-whisper), none (skip)",
    )
    parser.add_argument(
        "--tz",
        type=int,
        default=3,
        help="Timezone offset in hours for date arguments, e.g. 3 for Moscow (default: 3)",
    )
    return parser


def parse_duration(s: str) -> timedelta:
    units = {"h": "hours", "d": "days", "w": "weeks", "m": "minutes"}
    suffix = s[-1].lower()
    if suffix not in units:
        raise argparse.ArgumentTypeError(f"Unknown time unit '{suffix}'. Use h/d/w/m")
    try:
        value = int(s[:-1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid duration: {s}")
    return timedelta(**{units[suffix]: value})


def transcribe_file_openrouter(audio_path: str, audio_format: str = "ogg") -> str | None:
    if not OPENROUTER_API_KEY:
        print("  [warn] OPENROUTER_API_KEY not set, skipping transcription", file=sys.stderr)
        return None
    try:
        import base64
        from openai import OpenAI

        openrouter = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        response = openrouter.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": audio_format},
                        },
                        {
                            "type": "text",
                            "text": "Transcribe this audio message exactly. The language is Russian. Output only the transcription, nothing else.",
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [warn] OpenRouter transcription failed: {e}", file=sys.stderr)
        return None


_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("  Loading whisper model (first time may download ~1GB)...", file=sys.stderr)
        _whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_file_local(audio_path: str) -> str | None:
    try:
        model = get_whisper_model()
        segments, _ = model.transcribe(audio_path, language="ru")
        text = " ".join(seg.text.strip() for seg in segments)
        return text if text else None
    except Exception as e:
        print(f"  [warn] Local transcription failed: {e}", file=sys.stderr)
        return None


def transcribe_file(audio_path: str, method: str, audio_format: str = "ogg") -> str | None:
    if method == "openrouter":
        return transcribe_file_openrouter(audio_path, audio_format)
    elif method == "local":
        return transcribe_file_local(audio_path)
    return None


def extract_audio_from_video(video_path: str) -> str | None:
    import subprocess
    audio_path = video_path + ".ogg"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libopus", "-b:a", "32k",
                "-ar", "16000", "-ac", "1",
                audio_path,
            ],
            check=True,
            capture_output=True,
        )
        return audio_path
    except FileNotFoundError:
        print("  [warn] ffmpeg not found. Install with: brew install ffmpeg", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"  [warn] ffmpeg failed: {e.stderr.decode()[:200]}", file=sys.stderr)
        return None


async def transcribe_audio(client: TelegramClient, message, method: str) -> str | None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            tmp_path = f.name
        await client.download_media(message, file=tmp_path)
        return transcribe_file(tmp_path, method, audio_format="ogg")
    except Exception as e:
        print(f"  [warn] Audio transcription failed for msg {message.id}: {e}", file=sys.stderr)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def transcribe_video(client: TelegramClient, message, method: str) -> str | None:
    video_path = None
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
        await client.download_media(message, file=video_path)

        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            return None
        return transcribe_file(audio_path, method, audio_format="ogg")
    except Exception as e:
        print(f"  [warn] Video transcription failed for msg {message.id}: {e}", file=sys.stderr)
        return None
    finally:
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


async def download_image(client: TelegramClient, message, images_dir: str) -> str | None:
    try:
        filename = f"msg_{message.id}.jpg"
        filepath = os.path.join(images_dir, filename)
        await client.download_media(message, file=filepath)
        return filepath
    except Exception as e:
        print(f"  [warn] Image download failed for msg {message.id}: {e}", file=sys.stderr)
        return None


async def describe_image_openrouter(image_path: str) -> str | None:
    if not OPENROUTER_API_KEY:
        print("  [warn] OPENROUTER_API_KEY not set, skipping image description", file=sys.stderr)
        return None
    try:
        import base64
        from openai import OpenAI

        openrouter = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        response = openrouter.chat.completions.create(
            model="google/gemma-3-4b-it:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Опиши это изображение подробно на русском языке. Укажи все важные детали: объекты, текст, людей, действия.",
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [warn] Image description failed: {e}", file=sys.stderr)
        return None


async def describe_image_local(image_path: str) -> str | None:
    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": "Опиши это изображение подробно на русском языке. Укажи все важные детали: объекты, текст, людей, действия.",
                "images": [_image_to_base64(image_path)],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"  [warn] Local image description failed (is ollama running with llava?): {e}", file=sys.stderr)
        return None


def _image_to_base64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


async def describe_image(image_path: str, method: str) -> str | None:
    if method == "openrouter":
        return await describe_image_openrouter(image_path)
    elif method == "local":
        return await describe_image_local(image_path)
    return None


def get_media_info(message) -> dict | None:
    media = message.media
    if media is None:
        return None

    if isinstance(media, MessageMediaPhoto):
        return {"type": "photo"}

    if isinstance(media, MessageMediaDocument):
        doc = media.document
        if doc is None:
            return {"type": "document"}

        info = {
            "type": "document",
            "mime_type": doc.mime_type,
            "size_bytes": doc.size,
        }

        for attr in doc.attributes:
            if isinstance(attr, DocumentAttributeAudio):
                info["type"] = "voice_message" if attr.voice else "audio"
                info["duration_seconds"] = attr.duration
                if attr.title:
                    info["title"] = attr.title
                if attr.performer:
                    info["performer"] = attr.performer
            elif isinstance(attr, DocumentAttributeVideo):
                info["type"] = "video_note" if attr.round_message else "video"
                info["duration_seconds"] = attr.duration
                info["width"] = attr.w
                info["height"] = attr.h
            elif isinstance(attr, DocumentAttributeSticker):
                info["type"] = "sticker"
                info["emoji"] = attr.alt
            elif isinstance(attr, DocumentAttributeFilename):
                info["filename"] = attr.file_name

        return info

    if isinstance(media, MessageMediaGeo):
        geo = media.geo
        return {"type": "geo", "lat": geo.lat, "long": geo.long}

    if isinstance(media, MessageMediaContact):
        return {
            "type": "contact",
            "phone": media.phone_number,
            "first_name": media.first_name,
            "last_name": media.last_name,
        }

    if isinstance(media, MessageMediaPoll):
        poll = media.poll
        return {
            "type": "poll",
            "question": poll.question.text if hasattr(poll.question, 'text') else str(poll.question),
            "answers": [a.text.text if hasattr(a.text, 'text') else str(a.text) for a in poll.answers],
        }

    if isinstance(media, MessageMediaWebPage):
        wp = media.webpage
        if hasattr(wp, "url"):
            return {"type": "webpage", "url": wp.url, "title": getattr(wp, "title", None)}
        return {"type": "webpage"}

    return {"type": type(media).__name__}


def get_forward_info(message) -> dict | None:
    fwd = message.forward
    if fwd is None:
        return None
    info = {"is_forwarded": True}
    if fwd.from_id:
        if isinstance(fwd.from_id, PeerUser):
            info["original_sender_id"] = fwd.from_id.user_id
        elif isinstance(fwd.from_id, PeerChannel):
            info["original_channel_id"] = fwd.from_id.channel_id
        elif isinstance(fwd.from_id, PeerChat):
            info["original_chat_id"] = fwd.from_id.chat_id
    if fwd.from_name:
        info["original_sender_name"] = fwd.from_name
    if fwd.date:
        info["original_date"] = fwd.date.isoformat()
    if fwd.channel_post:
        info["original_post_id"] = fwd.channel_post
    return info


async def resolve_sender(message, user_cache: dict) -> dict:
    sender = await message.get_sender()
    if sender is None:
        return {"id": message.sender_id, "name": "Unknown"}

    uid = sender.id
    if uid in user_cache:
        return user_cache[uid]

    info = {"id": uid}
    if hasattr(sender, "first_name"):
        info["first_name"] = sender.first_name
        info["last_name"] = sender.last_name
        info["username"] = sender.username
        info["is_bot"] = sender.bot
        info["name"] = f"{sender.first_name or ''} {sender.last_name or ''}".strip()
    elif hasattr(sender, "title"):
        info["name"] = sender.title
        info["is_channel"] = True
    else:
        info["name"] = str(uid)

    user_cache[uid] = info
    return info


async def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    user_tz = timezone(timedelta(hours=args.tz))
    now = datetime.now(user_tz)

    if args.last:
        delta = parse_duration(args.last)
        date_from = now - delta
        date_to = now
    else:
        date_from = args.from_date.replace(tzinfo=user_tz)
        date_to = args.to_date.replace(tzinfo=user_tz) if args.to_date else now
        if args.to_date:
            date_to = date_to + timedelta(days=1)

    if date_from >= date_to:
        print("Error: start date must be before end date", file=sys.stderr)
        sys.exit(1)

    chat_target = args.chat

    print(f"Connecting to Telegram...")
    client = TelegramClient("session", API_ID, API_HASH)
    await client.start(phone=PHONE)

    print(f"Resolving chat: {chat_target}")
    try:
        chat_id = int(chat_target)
        entity = await client.get_entity(chat_id)
    except ValueError:
        entity = await client.get_entity(chat_target)

    chat_title = getattr(entity, "title", getattr(entity, "username", str(chat_target)))
    print(f"Chat: {chat_title}")
    print(f"Period: {date_from.strftime('%Y-%m-%d %H:%M')} → {date_to.strftime('%Y-%m-%d %H:%M')}")

    images_dir = None
    if args.transcribe != "none":
        safe_name = "".join(c if c.isalnum() else "_" for c in chat_title)
        images_dir = f"images_{safe_name}"
        os.makedirs(images_dir, exist_ok=True)

    messages = []
    user_cache = {}
    count = 0

    first_seen = True
    async for message in client.iter_messages(entity, offset_date=date_to):
        if first_seen:
            print(f"  First message seen: id={message.id} date={message.date.isoformat()}", file=sys.stderr)
            first_seen = False
        if message.date < date_from:
            break
        if message.date >= date_to:
            continue

        count += 1
        if count % 100 == 0:
            print(f"  Processed {count} messages...", file=sys.stderr)

        sender = await resolve_sender(message, user_cache)
        media_info = get_media_info(message)
        forward_info = get_forward_info(message)

        entry = {
            "id": message.id,
            "date": message.date.isoformat(),
            "sender": sender,
            "text": message.text or "",
            "is_reply": message.reply_to is not None,
            "reply_to_msg_id": message.reply_to.reply_to_msg_id if message.reply_to else None,
        }

        if forward_info:
            entry["forward"] = forward_info

        if media_info:
            entry["media"] = media_info

        is_voice = media_info and media_info.get("type") in ("voice_message", "audio")
        if is_voice and args.transcribe != "none":
            print(f"  Transcribing audio message {message.id} ({args.transcribe})...")
            transcript = await transcribe_audio(client, message, args.transcribe)
            if transcript:
                entry["audio_transcript"] = transcript

        is_video = media_info and media_info.get("type") in ("video_note", "video")
        if is_video and args.transcribe != "none":
            print(f"  Transcribing video message {message.id} ({args.transcribe})...")
            transcript = await transcribe_video(client, message, args.transcribe)
            if transcript:
                entry["audio_transcript"] = transcript

        is_photo = media_info and media_info.get("type") == "photo"
        if is_photo and images_dir:
            print(f"  Downloading image from message {message.id}...")
            img_path = await download_image(client, message, images_dir)
            if img_path:
                entry["image_path"] = img_path
                if args.transcribe != "none":
                    print(f"  Describing image {message.id} ({args.transcribe})...")
                    description = await describe_image(img_path, args.transcribe)
                    if description:
                        entry["image_description"] = description

        if message.entities:
            entities = []
            for ent in message.entities:
                ent_data = {
                    "type": type(ent).__name__,
                    "offset": ent.offset,
                    "length": ent.length,
                }
                if hasattr(ent, "url") and ent.url:
                    ent_data["url"] = ent.url
                if hasattr(ent, "user_id") and ent.user_id:
                    ent_data["user_id"] = ent.user_id
                entities.append(ent_data)
            entry["entities"] = entities

        if message.reactions:
            try:
                reactions = []
                for r in message.reactions.results:
                    emoticon = getattr(r.reaction, "emoticon", None) or str(r.reaction)
                    reactions.append({"emoji": emoticon, "count": r.count})
                entry["reactions"] = reactions
            except Exception:
                pass

        if message.views is not None:
            entry["views"] = message.views

        if message.edit_date:
            entry["edit_date"] = message.edit_date.isoformat()

        if message.pinned:
            entry["pinned"] = True

        if message.grouped_id:
            entry["grouped_id"] = message.grouped_id

        messages.append(entry)

    messages.reverse()
    await client.disconnect()

    output_path = args.output
    if not output_path:
        safe_name = "".join(c if c.isalnum() else "_" for c in chat_title)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"chat_{safe_name}_{ts}.json"

    output = {
        "chat": {
            "id": entity.id,
            "title": chat_title,
        },
        "parse_period": {
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
        },
        "total_messages": len(messages),
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDone! {len(messages)} messages saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
