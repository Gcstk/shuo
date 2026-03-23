"""Remote SoulX-Duplug adapter with a Flux-compatible interface."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

import numpy as np
import websockets

from ..audio import b64encode_bytes, mulaw_to_pcm16, pcm16_resample
from ..log import ServiceLogger, env_flag
from .flux import FluxTurnInfo

log = ServiceLogger("Duplug")


class DuplugService:
    """
    WebSocket client adapter for the SoulX-Duplug server.

    The remote server input is:
    {
        "type": "audio",
        "session_id": "...",
        "audio": "<base64 float32 pcm>"
    }

    The remote server output is:
    {
        "type": "turn_state",
        "session_id": "...",
        "state": {
            "state": "blank|idle|nonidle|speak",
            "text": "...",
            "asr_segment": "...",
            "asr_buffer": "..."
        }
    }

    This adapter maps those outputs onto the Flux-style callbacks expected by shuo.
    """

    def __init__(
        self,
        on_end_of_turn: Callable[[str], Awaitable[None]],
        on_start_of_turn: Callable[[], Awaitable[None]],
        on_interim: Optional[Callable[[str], Awaitable[None]]] = None,
        on_turn_info: Optional[Callable[[FluxTurnInfo], Awaitable[None]]] = None,
    ):
        self._on_end_of_turn = on_end_of_turn
        self._on_start_of_turn = on_start_of_turn
        self._on_interim = on_interim
        self._on_turn_info = on_turn_info

        self._url = os.getenv("DUPLUG_WS_URL", "").strip()
        self._timeout = float(os.getenv("DUPLUG_WS_TIMEOUT", "10"))
        self._debug_mode = env_flag("DEBUG_MODE")

        self._ws: Optional[Any] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False
        self._encoding = "linear16"
        self._sample_rate = 16000
        self._session_id = f"shuo-{uuid.uuid4().hex}"
        self._session_started_at = 0.0
        self._duration_queue: asyncio.Queue[float] = asyncio.Queue()

        self._turn_started = False
        self._turn_index = 0
        self._current_turn_index: Optional[int] = None
        self._current_turn_audio_sec = 0.0
        self._last_interim = ""

    @property
    def is_active(self) -> bool:
        return self._running and self._ws is not None

    async def start(
        self,
        encoding: str = "linear16",
        sample_rate: int = 16000,
    ) -> None:
        if self._running:
            return
        if not self._url:
            raise ValueError("Missing DUPLUG_WS_URL")

        self._encoding = encoding
        self._sample_rate = sample_rate
        self._turn_started = False
        self._turn_index = 0
        self._current_turn_index = None
        self._current_turn_audio_sec = 0.0
        self._last_interim = ""
        self._session_id = f"shuo-{uuid.uuid4().hex}"
        self._session_started_at = time.monotonic()
        self._duration_queue = asyncio.Queue()

        self._ws = await websockets.connect(self._url, open_timeout=self._timeout)
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        log.connected()

    async def send(self, audio_bytes: bytes) -> None:
        if not self._running or not self._ws:
            return

        try:
            samples = self._decode_audio(audio_bytes)
        except Exception as exc:
            log.error("Decode failed", exc)
            return

        if samples.size == 0:
            return

        payload = {
            "type": "audio",
            "session_id": self._session_id,
            "audio": b64encode_bytes(samples.astype(np.float32).tobytes()),
        }

        try:
            await self._ws.send(json.dumps(payload))
            self._duration_queue.put_nowait(len(samples) / 16000.0)
        except Exception as exc:
            log.error("Send failed", exc)

    async def stop(self) -> None:
        self._running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._turn_started = False
        self._current_turn_index = None
        self._current_turn_audio_sec = 0.0
        self._last_interim = ""
        log.disconnected()

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        encoding = self._encoding.lower()
        if encoding == "mulaw":
            pcm16 = mulaw_to_pcm16(audio_bytes)
            sample_rate = 8000
        elif encoding in {"linear16", "pcm16"}:
            pcm16 = audio_bytes
            sample_rate = self._sample_rate
        else:
            raise ValueError(f"unsupported encoding: {self._encoding}")

        pcm16 = pcm16_resample(pcm16, sample_rate, 16000)
        if not pcm16:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

    async def _receive_loop(self) -> None:
        try:
            while self._running and self._ws:
                raw = await self._ws.recv()
                await self._handle_message(raw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._running:
                log.error("Receive failed", exc)
        finally:
            self._running = False

    async def _handle_message(self, raw: str) -> None:
        self._log_incoming_raw(raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.error(f"Invalid JSON: {raw[:120]}")
            return

        if data.get("type") != "turn_state":
            return

        try:
            chunk_duration_sec = self._duration_queue.get_nowait()
        except asyncio.QueueEmpty:
            chunk_duration_sec = 0.0

        self._log_turn_state(data, chunk_duration_sec)
        await self._process_turn_state(data.get("state") or {}, chunk_duration_sec)

    async def _process_turn_state(
        self,
        result: dict,
        chunk_duration_sec: float,
    ) -> None:
        state = (result.get("state") or "").strip().lower()
        if state in {"", "blank"}:
            return

        if state == "nonidle":
            await self._handle_nonidle(result, chunk_duration_sec)
            return

        if state == "speak":
            await self._handle_speak(result, chunk_duration_sec)
            return

        if state == "idle" and self._turn_started:
            await self._emit_turn_info(
                event="Update",
                transcript=self._best_interim_text(result),
                audio_window_end=self._current_turn_audio_sec,
            )

    async def _handle_nonidle(self, result: dict, chunk_duration_sec: float) -> None:
        transcript = self._best_interim_text(result)
        if not self._turn_started:
            self._turn_started = True
            self._turn_index += 1
            self._current_turn_index = self._turn_index
            self._current_turn_audio_sec = chunk_duration_sec
            await self._emit_turn_info(
                event="StartOfTurn",
                transcript=transcript,
                audio_window_end=self._current_turn_audio_sec,
            )
            await self._on_start_of_turn()
        else:
            self._current_turn_audio_sec += chunk_duration_sec
            await self._emit_turn_info(
                event="Update",
                transcript=transcript,
                audio_window_end=self._current_turn_audio_sec,
            )

        await self._emit_interim(transcript)

    async def _handle_speak(self, result: dict, chunk_duration_sec: float) -> None:
        final_text = self._best_final_text(result)
        if not self._turn_started:
            self._turn_started = True
            self._turn_index += 1
            self._current_turn_index = self._turn_index
            self._current_turn_audio_sec = chunk_duration_sec
            await self._emit_turn_info(
                event="StartOfTurn",
                transcript=final_text,
                audio_window_end=self._current_turn_audio_sec,
            )
            await self._on_start_of_turn()
        else:
            self._current_turn_audio_sec += chunk_duration_sec

        await self._emit_turn_info(
            event="EndOfTurn",
            transcript=final_text,
            audio_window_end=self._current_turn_audio_sec,
        )

        self._turn_started = False
        self._current_turn_index = None
        self._current_turn_audio_sec = 0.0
        self._last_interim = ""

        await self._on_end_of_turn(final_text.strip())

    def _best_interim_text(self, result: dict) -> str:
        return (
            (result.get("asr_buffer") or "").strip()
            or (result.get("asr_segment") or "").strip()
        )

    def _best_final_text(self, result: dict) -> str:
        return (
            (result.get("text") or "").strip()
            or self._best_interim_text(result)
            or self._last_interim
        )

    async def _emit_interim(self, transcript: str) -> None:
        if not self._on_interim:
            return
        text = (transcript or "").strip()
        if not text or text == self._last_interim:
            return
        self._last_interim = text
        await self._on_interim(text)

    async def _emit_turn_info(
        self,
        event: str,
        transcript: str,
        audio_window_end: float,
    ) -> None:
        if not self._on_turn_info or self._current_turn_index is None:
            return
        await self._on_turn_info(
            FluxTurnInfo(
                event=event,
                transcript=(transcript or "").strip(),
                turn_index=self._current_turn_index,
                audio_window_start=0.0,
                audio_window_end=max(audio_window_end, 0.0),
                received_at=time.monotonic(),
            )
        )

    def _elapsed_ms(self) -> float:
        if self._session_started_at <= 0:
            return 0.0
        return (time.monotonic() - self._session_started_at) * 1000.0

    def _log_incoming_raw(self, raw: str) -> None:
        if not self._debug_mode:
            return
        log.info(f"recv +{self._elapsed_ms():.1f}ms raw={raw}")

    def _log_turn_state(self, data: dict, chunk_duration_sec: float) -> None:
        if not self._debug_mode:
            return

        state = data.get("state") or {}
        label = (state.get("state") or "").strip().lower() or "unknown"
        text = (state.get("text") or "").strip()
        asr_segment = (state.get("asr_segment") or "").strip()
        asr_buffer = (state.get("asr_buffer") or "").strip()

        def _clip(value: str, limit: int = 80) -> str:
            if len(value) <= limit:
                return value
            return value[: limit - 1] + "…"

        log.info(
            "recv "
            f"+{self._elapsed_ms():.1f}ms "
            f"state={label} "
            f"chunk={chunk_duration_sec * 1000:.1f}ms "
            f"turn_started={self._turn_started} "
            f"turn_index={self._current_turn_index} "
            f"text={_clip(text)!r} "
            f"segment={_clip(asr_segment)!r} "
            f"buffer={_clip(asr_buffer)!r}"
        )
