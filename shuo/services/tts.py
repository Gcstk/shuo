"""Provider-aware streaming TTS service."""

import os
import json
import asyncio
from typing import Optional, Callable, Awaitable

import websockets
from websockets.client import WebSocketClientProtocol

from ..audio import b64decode_bytes, b64encode_bytes, pcm16_to_mulaw
from ..log import ServiceLogger

log = ServiceLogger("TTS")


class TTSService:
    """
    流式 TTS 服务。

    对上层暴露统一接口：
    - send(text): 发送文本增量
    - flush(): 通知一轮文本发送完成
    - on_audio(audio_b64): 回调统一输出 ulaw_8000 base64 音频
    """

    def __init__(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ):
        self._on_audio = on_audio
        self._on_done = on_done

        self._ws: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False

        self._provider = os.getenv("TTS_PROVIDER", "elevenlabs").strip().lower()

        self._api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

        self._dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "sk-2c31a3a763e14c48a35a7c1678fa0c5c")
        self._qwen_model = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-flash-realtime-2025-11-27")
        self._qwen_voice = os.getenv("QWEN_TTS_VOICE", "Cherry")
        self._qwen_language = os.getenv("QWEN_TTS_LANGUAGE", "Chinese")
        self._qwen_mode = os.getenv("QWEN_TTS_MODE", "commit")
        self._qwen_sample_rate = int(os.getenv("QWEN_TTS_SAMPLE_RATE", "24000"))

    @property
    def is_active(self) -> bool:
        return self._running and self._ws is not None

    def bind(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> None:
        self._on_audio = on_audio
        self._on_done = on_done

    async def start(self) -> None:
        if self._running:
            return

        if self._provider == "qwen":
            await self._start_qwen()
            return

        await self._start_elevenlabs()

    async def send(self, text: str) -> None:
        if not self._ws or not self._running:
            return

        try:
            if self._provider == "qwen":
                await self._ws.send(json.dumps({
                    "type": "input_text_buffer.append",
                    "text": text,
                }))
            else:
                await self._ws.send(json.dumps({
                    "text": text,
                    "try_trigger_generation": True,
                }))
        except Exception as exc:
            log.error("Send failed", exc)

    async def flush(self) -> None:
        if not self._ws or not self._running:
            return

        try:
            if self._provider == "qwen":
                # Qwen 没有名为 flush 的事件，但官方协议支持
                # input_text_buffer.commit 来立即合成当前缓冲文本。
                await self._ws.send(json.dumps({"type": "input_text_buffer.commit"}))
            else:
                await self._ws.send(json.dumps({
                    "text": "",
                    "flush": True,
                }))
        except Exception as exc:
            log.error("Flush failed", exc)

    async def stop(self) -> None:
        if not self._running:
            return

        try:
            if self._provider == "qwen" and self._ws:
                await self._ws.send(json.dumps({"type": "session.finish"}))
                await asyncio.sleep(0.1)
            else:
                await self.flush()
                await asyncio.sleep(0.2)
        except Exception as exc:
            log.error("Stop failed", exc)
        finally:
            await self._cleanup()

        log.disconnected()

    async def cancel(self) -> None:
        self._running = False
        await self._cleanup()
        log.cancelled()

    async def _start_elevenlabs(self) -> None:
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input?"
            f"model_id=eleven_turbo_v2_5&"
            f"output_format=ulaw_8000"
        )

        try:
            self._ws = await websockets.connect(url)
            self._running = True

            resp = getattr(self._ws, "response", None)
            hdrs = getattr(resp, "headers", {}) if resp else {}
            region = hdrs.get("x-region", "unknown")
            log.info(f"Region: {region}")

            await self._ws.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
                "xi_api_key": self._api_key,
            }))

            self._receive_task = asyncio.create_task(self._receive_loop())
            log.connected()
        except Exception as exc:
            log.error("Connection failed", exc)
            raise

    async def _start_qwen(self) -> None:
        if not self._dashscope_api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY for Qwen TTS")

        url = (
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
            f"?model={self._qwen_model}"
        )

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {self._dashscope_api_key}"},
            )
            self._running = True
            await self._ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "mode": self._qwen_mode,
                    "voice": self._qwen_voice,
                    "language_type": self._qwen_language,
                    "response_format": "pcm",
                    "sample_rate": self._qwen_sample_rate,
                },
            }))
            self._receive_task = asyncio.create_task(self._receive_loop())
            log.connected()
        except Exception as exc:
            log.error("Connection failed", exc)
            raise

    async def _cleanup(self) -> None:
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

    async def _receive_loop(self) -> None:
        try:
            while self._running and self._ws:
                try:
                    message = await self._ws.recv()
                    await self._handle_message(message)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as exc:
                    log.error("Receive failed", exc)
                    break
        finally:
            if self._running:
                self._running = False
                await self._on_done()

    async def _handle_message(self, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            log.error(f"Invalid JSON: {message[:100]}")
            return

        if self._provider == "qwen":
            await self._handle_qwen_message(data)
            return

        if "audio" in data and data["audio"]:
            await self._on_audio(data["audio"])

        if data.get("isFinal", False):
            await self._on_done()

    async def _handle_qwen_message(self, data: dict) -> None:
        event_type = data.get("type")
        if event_type == "response.audio.delta":
            audio_bytes = pcm16_to_mulaw(
                b64decode_bytes(data.get("delta", "")),
                self._qwen_sample_rate,
            )
            await self._on_audio(b64encode_bytes(audio_bytes))
            return

        if event_type == "response.done":
            await self._on_done()
            return

        if event_type == "error":
            log.error(f"Qwen TTS error: {data}")
