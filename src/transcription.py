"""Audio + video transcription for incoming Telegram messages."""

import asyncio
import os
import subprocess
import sys
import tempfile

from src.config import OPENROUTER_API_KEY


_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("  Loading whisper model (first time may download ~1GB)...", file=sys.stderr)
        _whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_file_local(audio_path: str) -> str | None:
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(audio_path, language="ru")
        text = " ".join(seg.text.strip() for seg in segments)
        return text or None
    except Exception as e:
        print(f"  [warn] local transcription failed: {e}", file=sys.stderr)
        return None


def transcribe_file_openrouter(audio_path: str, audio_format: str = "ogg") -> str | None:
    if not OPENROUTER_API_KEY:
        return None
    try:
        import base64
        from openai import OpenAI

        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        resp = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
                    {"type": "text", "text": "Transcribe this audio exactly. Language is Russian. Output only the transcription."},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [warn] OpenRouter transcription failed: {e}", file=sys.stderr)
        return None


def transcribe_file(path: str, method: str, audio_format: str = "ogg") -> str | None:
    if method == "local":
        return transcribe_file_local(path)
    if method == "openrouter":
        return transcribe_file_openrouter(path, audio_format)
    return None


def extract_audio_from_video(video_path: str) -> str | None:
    audio_path = video_path + ".ogg"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libopus", "-b:a", "32k",
                "-ar", "16000", "-ac", "1",
                audio_path,
            ],
            check=True, capture_output=True,
        )
        return audio_path
    except FileNotFoundError:
        print("  [warn] ffmpeg not found. Install with: brew install ffmpeg", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"  [warn] ffmpeg failed: {e.stderr.decode()[:200]}", file=sys.stderr)
        return None


async def transcribe_voice_message(telethon_client, message, method: str) -> str | None:
    """Download a Telethon voice message and transcribe it."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            tmp = f.name
        await telethon_client.download_media(message, file=tmp)
        return await asyncio.to_thread(transcribe_file, tmp, method, "ogg")
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


async def transcribe_video_message(telethon_client, message, method: str) -> str | None:
    """Download a Telethon video/video_note, extract audio, transcribe."""
    video_tmp = None
    audio_tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_tmp = f.name
        await telethon_client.download_media(message, file=video_tmp)
        audio_tmp = await asyncio.to_thread(extract_audio_from_video, video_tmp)
        if not audio_tmp:
            return None
        return await asyncio.to_thread(transcribe_file, audio_tmp, method, "ogg")
    finally:
        for p in (video_tmp, audio_tmp):
            if p and os.path.exists(p):
                os.unlink(p)
