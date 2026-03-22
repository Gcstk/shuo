"""
Deepgram Flux service -- always-on STT + turn detection.

A single persistent WebSocket to Deepgram using the v2 listen API.
Receives transport audio continuously and emits turn events.

Replaces both local VAD (Silero) and separate STT (Deepgram v1).

教学重点：
1. 本服务持有一条长期 WebSocket 连接
2. 上行音频持续送入这条连接
3. Flux 实时回推 TurnInfo（StartOfTurn/Update/EndOfTurn）
4. EndOfTurn 会带 transcript，作为“触发回答”的输入

因此，轮次检测的复杂度被外部服务吸收，本地状态机只需消费事件。
"""

import os
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

try:
    from deepgram import AsyncDeepgramClient, DeepgramClientEnvironment
except ImportError:  # pragma: no cover - handled at runtime in start()
    AsyncDeepgramClient = None
    DeepgramClientEnvironment = None

from ..log import ServiceLogger

log = ServiceLogger("Flux")


@dataclass(frozen=True)
class FluxTurnInfo:
    """把 Flux TurnInfo 归一化，供会话层做 trace 和统计。"""
    event: str
    transcript: str
    turn_index: Optional[int]
    audio_window_start: float
    audio_window_end: float
    received_at: float


class FluxService:
    """
    Deepgram Flux 流式识别服务。

    约束：
    - 音频格式由 transport 决定（Twilio: mulaw/8k, Browser: linear16/16k）
    - turn 事件由服务端给出（StartOfTurn / Update / EndOfTurn）
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

        self._api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self._client: Optional[AsyncDeepgramClient] = None
        self._connection = None
        self._cm = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False
        self._turn_started = False
        self._last_interim = ""

    @property
    def is_active(self) -> bool:
        return self._running and self._connection is not None

    async def start(
        self,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
    ) -> None:
        """
        建立 Flux 长连接（每通电话生命周期内常驻）。

        一旦启动，后续所有上行音频都通过 send() 持续送入。
        """
        if self._running:
            return
        if AsyncDeepgramClient is None or DeepgramClientEnvironment is None:
            raise RuntimeError(
                "deepgram SDK is not installed; install project dependencies first"
            )

        try:
            deepgram_eu = DeepgramClientEnvironment(
                base="wss://api.eu.deepgram.com",
                production="wss://api.eu.deepgram.com",
                agent="wss://agent.eu.deepgram.com",
            )
            self._client = AsyncDeepgramClient(
                api_key=self._api_key,
                environment=deepgram_eu,
            )

            # listen.v2.connect 返回异步上下文管理器，这里手动 enter/exit。
            self._cm = self._client.listen.v2.connect(
                model="flux-general-en",
                encoding=encoding,
                sample_rate=sample_rate,
            )
            self._connection = await self._cm.__aenter__()

            # 注册消息与错误回调，由 SDK 在后台线程/任务触发。
            self._connection.on("message", self._on_message)
            self._connection.on("Error", self._on_error)

            # start_listening 持续拉取服务端消息，不阻塞主协程。
            self._listener_task = asyncio.create_task(
                self._connection.start_listening()
            )

            self._running = True
            log.connected()

        except Exception as e:
            log.error("Connection failed", e)
            await self._cleanup()
            raise

    async def send(self, audio_bytes: bytes) -> None:
        """发送一帧上行音频给 Flux。"""
        if not self._connection or not self._running:
            return

        try:
            await self._connection.send_media(audio_bytes)
        except Exception as e:
            log.error("Send failed", e)

    async def stop(self) -> None:
        """断开 Flux 并回收资源。"""
        self._running = False
        await self._cleanup()
        log.disconnected()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False
        self._turn_started = False
        self._last_interim = ""

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None

        self._connection = None
        self._client = None

    async def _on_message(self, message, *args, **kwargs) -> None:
        """
        处理 Flux 返回消息。

        重点看 TurnInfo：
        - StartOfTurn: 用户开始说话（可用于打断）
        - EndOfTurn:   用户说完一轮，附带 transcript
        """
        try:
            msg_type = getattr(message, "type", None)

            if msg_type == "TurnInfo":
                event = getattr(message, "event", None)
                transcript = self._extract_turn_transcript(message)
                # 保留 Deepgram turn metadata，业务层可以据此估算
                # 用户开口、首个 interim、最终 transcript 的时间点。
                turn_info = FluxTurnInfo(
                    event=event or "",
                    transcript=transcript,
                    turn_index=self._extract_turn_index(message),
                    audio_window_start=self._extract_audio_window_value(
                        message,
                        "audio_window_start",
                    ),
                    audio_window_end=self._extract_audio_window_value(
                        message,
                        "audio_window_end",
                    ),
                    received_at=time.monotonic(),
                )

                if self._on_turn_info:
                    # 先把结构化 TurnInfo 交给上层做 trace，
                    # 再继续走原有的状态机/字幕回调逻辑。
                    await self._on_turn_info(turn_info)

                if event == "EndOfTurn":
                    self._turn_started = False
                    self._last_interim = ""
                    await self._on_end_of_turn(transcript.strip())

                elif event == "StartOfTurn":
                    self._turn_started = True
                    await self._on_start_of_turn()
                    await self._emit_interim(transcript)

                elif event in {"Update", "TurnResumed", "EagerEndOfTurn"}:
                    if transcript and not self._turn_started:
                        self._turn_started = True
                        await self._on_start_of_turn()
                    await self._emit_interim(transcript)

            elif msg_type == "Results" and self._on_interim:
                channel = getattr(message, "channel", None)
                if channel:
                    alternatives = getattr(channel, "alternatives", None)
                    if alternatives:
                        alt = (
                            alternatives[0]
                            if isinstance(alternatives, list)
                            else alternatives
                        )
                        transcript = getattr(alt, "transcript", "")
                        await self._emit_interim(transcript)

        except Exception as e:
            log.error("Message handling failed", e)

    @staticmethod
    def _extract_turn_transcript(message) -> str:
        transcript = getattr(message, "transcript", "") or ""
        if transcript:
            return transcript.strip()

        metadata = getattr(message, "metadata", None)
        if metadata:
            transcript = getattr(metadata, "transcript", "") or ""
            if transcript:
                return transcript.strip()

        return ""

    @staticmethod
    def _extract_turn_index(message) -> Optional[int]:
        # Deepgram SDK 不同对象上字段位置可能不同，这里统一兼容。
        value = getattr(message, "turn_index", None)
        if value is None:
            metadata = getattr(message, "metadata", None)
            value = getattr(metadata, "turn_index", None) if metadata else None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_audio_window_value(message, field_name: str) -> float:
        # audio_window_* 由 Flux 给出，可近似恢复“这轮用户语音”
        # 在服务端时间轴上的开始/结束位置。
        value = getattr(message, field_name, None)
        if value is None:
            metadata = getattr(message, "metadata", None)
            value = getattr(metadata, field_name, None) if metadata else None
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    async def _emit_interim(self, transcript: str) -> None:
        if not self._on_interim:
            return
        text = transcript.strip()
        # UI 只需要看到文本推进，不需要重复刷同一份 transcript。
        if not text or text == self._last_interim:
            return
        self._last_interim = text
        await self._on_interim(text)

    async def _on_error(self, error, *args, **kwargs) -> None:
        """Handle Deepgram errors."""
        log.error("Deepgram: " + str(error))
