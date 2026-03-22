"""Transport adapters for Twilio and browser sessions."""

from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import WebSocket

from .audio import (
    BROWSER_FLUX_SAMPLE_RATE,
    b64decode_bytes,
    b64encode_bytes,
    mulaw_to_pcm16,
    pcm16_resample,
)
from .types import Event, MediaEvent, Phase, StreamStartEvent, StreamStopEvent
from .services.twilio_client import parse_twilio_message
from .log import get_logger

logger = get_logger("shuo.transport")


class BaseTransport:
    supports_live_text = False
    flux_encoding = "mulaw"
    flux_sample_rate = 8000

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def receive_event(self) -> Optional[Event]:
        raise NotImplementedError

    async def send_audio_chunk(self, audio_base64: str) -> None:
        raise NotImplementedError

    async def clear_audio(self) -> None:
        raise NotImplementedError

    async def send_transcript(self, speaker: str, text: str, final: bool) -> None:
        return None

    async def send_phase(self, phase: Phase) -> None:
        return None

    async def send_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None


class TwilioTransport(BaseTransport):
    def __init__(self, websocket: WebSocket):
        super().__init__(websocket)
        self._stream_sid = ""

    async def receive_event(self) -> Optional[Event]:
        raw = await self.websocket.receive_text()
        data = json.loads(raw)
        event = parse_twilio_message(data)
        if isinstance(event, StreamStartEvent):
            self._stream_sid = event.stream_sid
        return event

    async def send_audio_chunk(self, audio_base64: str) -> None:
        if not self._stream_sid:
            return
        await self.websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {"payload": audio_base64},
        }))

    async def clear_audio(self) -> None:
        if not self._stream_sid:
            return
        await self.websocket.send_text(json.dumps({
            "event": "clear",
            "streamSid": self._stream_sid,
        }))


class BrowserTransport(BaseTransport):
    supports_live_text = True
    flux_encoding = "linear16"
    flux_sample_rate = BROWSER_FLUX_SAMPLE_RATE

    def __init__(self, websocket: WebSocket):
        super().__init__(websocket)
        self._started = False
        self._input_sample_rate = 16000
        self._stream_sid = f"browser-{uuid.uuid4().hex[:12]}"

    async def send_ready(self) -> None:
        await self.websocket.send_text(json.dumps({
            "type": "ready",
            "stream_id": self._stream_sid,
        }))

    async def receive_event(self) -> Optional[Event]:
        try:
            raw = await self.websocket.receive_text()
        except Exception:
            return StreamStopEvent()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error("invalid JSON message")
            return None

        msg_type = data.get("type")
        if msg_type == "start":
            self._input_sample_rate = int(data.get("sample_rate", self._input_sample_rate))
            self._started = True
            return StreamStartEvent(stream_sid=self._stream_sid)

        if msg_type == "audio":
            if not self._started:
                await self._send_error("session not started")
                return None
            if data.get("encoding") != "pcm16":
                await self._send_error("audio encoding must be pcm16")
                return None
            sample_rate = int(data.get("sample_rate", self._input_sample_rate))
            try:
                pcm_bytes = b64decode_bytes(data.get("audio_b64", ""))
                audio_bytes = pcm16_resample(
                    pcm_bytes,
                    sample_rate,
                    self.flux_sample_rate,
                )
            except Exception as exc:
                await self._send_error(f"invalid audio payload: {exc}")
                return None
            return MediaEvent(audio_bytes=audio_bytes)

        if msg_type == "stop":
            return StreamStopEvent()

        await self._send_error(f"unsupported message type: {msg_type}")
        return None

    async def send_audio_chunk(self, audio_base64: str) -> None:
        pcm_bytes = mulaw_to_pcm16(b64decode_bytes(audio_base64))
        await self.websocket.send_text(json.dumps({
            "type": "audio",
            "encoding": "pcm16",
            "sample_rate": 8000,
            "audio_b64": b64encode_bytes(pcm_bytes),
        }))

    async def clear_audio(self) -> None:
        await self.websocket.send_text(json.dumps({"type": "clear"}))

    async def send_transcript(self, speaker: str, text: str, final: bool) -> None:
        await self.websocket.send_text(json.dumps({
            "type": "transcript",
            "speaker": speaker,
            "text": text,
            "final": final,
        }))

    async def send_phase(self, phase: Phase) -> None:
        await self.websocket.send_text(json.dumps({
            "type": "state",
            "phase": phase.name,
        }))

    async def _send_error(self, message: str) -> None:
        logger.warning(f"Browser transport error: {message}")
        await self.websocket.send_text(json.dumps({
            "type": "error",
            "message": message,
        }))
