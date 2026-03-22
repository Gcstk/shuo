"""
Agent -- self-contained LLM -> TTS -> Player pipeline.

Encapsulates the entire agent response lifecycle.
Owns conversation history across turns.

    start_turn(transcript) -> add to history -> LLM -> TTS -> Player -> transport
    cancel_turn()          -> cancel all, keep history

TTS connections are managed by TTSPool (see services/tts_pool.py).

这个模块是“下行语音生成链路”的核心。
一次 Agent 回答的实时流水线如下：

    用户转写文本 -> LLM 流式 token ->
    TTS 流式音频 chunk -> Player 按节奏发回 transport

关键能力：
1. 边生成边合成边播放（不等整句）
2. 支持 barge-in 抢话时的整链路快速取消
3. 记录每轮关键延迟点（首 token、首音频、总时长）
"""

import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Optional

from .services.llm import LLMService
from .services.tts import TTSService
from .services.tts_pool import TTSPool
from .services.player import AudioPlayer
from .tracer import Tracer
from .log import ServiceLogger

log = ServiceLogger("Agent")


def _ms_since(t0: float) -> int:
    """Milliseconds elapsed since t0."""
    return int((time.monotonic() - t0) * 1000)


class Agent:
    """
    自包含的 Agent 回答流水线。

    设计说明：
    - LLMService 持久存在：跨轮保留会话历史
    - TTS 从连接池获取：降低每轮建连延迟
    - Player 每轮新建：简化播放生命周期管理
    """

    def __init__(
        self,
        send_audio: Callable[[str], Awaitable[None]],
        clear_audio: Callable[[], Awaitable[None]],
        on_done: Callable[[], None],
        tts_pool: TTSPool,
        tracer: Tracer,
        on_text: Optional[Callable[[str, bool], Awaitable[None]]] = None,
    ):
        self._send_audio = send_audio
        self._clear_audio = clear_audio
        self._on_done = on_done
        self._tts_pool = tts_pool
        self._tracer = tracer
        self._on_text = on_text

        # 持久 LLM：每轮新输入会追加到 history，实现多轮上下文。
        self._llm = LLMService(
            on_token=self._on_llm_token,
            on_done=self._on_llm_done,
        )

        # 轮次级组件：开始时创建/绑定，取消或结束后释放。
        self._tts: Optional[TTSService] = None
        self._player: Optional[AudioPlayer] = None
        self._active = False

        # Current turn number (for tracer)
        self._turn: int = 0

        # Latency milestones (monotonic timestamps, reset each turn)
        self._t0: float = 0.0
        self._t_tts_conn: float = 0.0
        self._t_first_token: float = 0.0
        self._t_first_audio: float = 0.0
        self._got_first_token = False
        self._got_first_audio = False
        self._assistant_text = ""

    @property
    def is_turn_active(self) -> bool:
        return self._active

    @property
    def history(self) -> List[Dict[str, str]]:
        """Read-only access to conversation history (owned by LLM)."""
        return self._llm.history

    # ── Turn Lifecycle ──────────────────────────────────────────────

    async def start_turn(
        self,
        transcript: str,
        trace_turn: Optional[int] = None,
    ) -> None:
        """
        启动一轮 Agent 回答。

        执行顺序：
        1. 拿到 TTS（优先 warm 连接）
        2. 创建播放器
        3. 启动 LLM 流式生成
        """
        if self._active:
            await self.cancel_turn()

        self._active = True
        self._got_first_token = False
        self._got_first_audio = False
        self._assistant_text = ""

        # 记录本轮 trace，后续用于可视化时序分析。
        if trace_turn and self._tracer.has_turn(trace_turn):
            # 复用会话层提前创建好的 turn，这样图上的起点就是“用户开口”，
            # 而不是“Agent 开始回答”。
            self._turn = trace_turn
            self._tracer.update_turn_transcript(self._turn, transcript)
        else:
            self._turn = self._tracer.begin_turn(transcript)
        self._t0 = self._tracer.get_turn_start(self._turn) or time.monotonic()
        self._tracer.begin(self._turn, "tts_pool")

        # 从池中拿 TTS：
        # - warm: 基本即时
        # - cold: 需要等待建连
        self._tts = await self._tts_pool.get(
            on_audio=self._on_tts_audio,
            on_done=self._on_tts_done,
        )
        self._t_tts_conn = time.monotonic()
        self._tracer.end(self._turn, "tts_pool")

        # Create player
        self._player = AudioPlayer(
            send_audio=self._send_audio,
            send_clear=self._clear_audio,
            on_done=self._on_playback_done,
        )

        # 启动 LLM 后，token 会通过 _on_llm_token 回调持续到达。
        self._tracer.begin(self._turn, "llm")
        await self._llm.start(transcript)

        tts_ms = int((self._t_tts_conn - self._t0) * 1000)
        log.info(f"Turn started  (TTS {tts_ms}ms = {tts_ms}ms setup)")

    async def cancel_turn(self) -> None:
        """
        取消当前轮次，保留历史。

        取消顺序固定为 LLM -> TTS -> Player：
        - 先停文本源头（LLM）
        - 再停语音合成（TTS）
        - 最后清播放缓冲（Player clear）
        这样能尽快止住“旧回答继续说话”。
        """
        if not self._active:
            return

        elapsed = _ms_since(self._t0) if self._t0 else 0
        self._active = False

        # Mark turn as cancelled (ends all open spans)
        self._tracer.cancel_turn(self._turn)

        # 取消顺序非常关键，避免下游继续消耗上游残留数据。
        await self._llm.cancel()

        if self._tts:
            await self._tts.cancel()
            self._tts = None

        if self._player:
            if self._player.is_playing:
                await self._player.stop_and_clear()
            self._player = None

        log.info(f"Turn cancelled at +{elapsed}ms (history preserved)")

    async def cleanup(self) -> None:
        """Final cleanup when call ends."""
        if self._active:
            await self.cancel_turn()

    # ── Internal Callbacks ──────────────────────────────────────────

    async def _on_llm_token(self, token: str) -> None:
        """
        LLM token 回调：把文本增量实时送给 TTS。

        这一步是“流式体验”的核心，不等待整段文本生成完毕。
        """
        if not self._active or not self._tts:
            return

        self._assistant_text += token
        if not self._got_first_token:
            self._got_first_token = True
            self._t_first_token = time.monotonic()
            self._tracer.mark(self._turn, "llm_first_token")
            self._tracer.begin(self._turn, "tts")
            log.info(f"⏱  LLM first token  +{_ms_since(self._t0)}ms")

        if self._on_text:
            await self._on_text(self._assistant_text, False)

        await self._tts.send(token)

    async def _on_llm_done(self) -> None:
        """LLM 结束后触发 TTS flush，催出尾段音频。"""
        if not self._active or not self._tts:
            return
        self._tracer.end(self._turn, "llm")
        await self._tts.flush()

    async def _on_tts_audio(self, audio_base64: str) -> None:
        """
        TTS 音频回调：把音频 chunk 交给播放器发往 Twilio。

        第一包音频到达时会记录“端到端首音频延迟”。
        """
        if not self._active or not self._player:
            return

        if not self._got_first_audio:
            self._got_first_audio = True
            self._t_first_audio = time.monotonic()
            self._tracer.mark(self._turn, "tts_first_audio")
            self._tracer.begin(self._turn, "player")
            ttft = _ms_since(self._t0)
            since_token = int((self._t_first_audio - self._t_first_token) * 1000) if self._got_first_token else 0
            log.info(f"⏱  TTS first audio  +{ttft}ms  (TTS latency {since_token}ms)")

        await self._player.send_chunk(audio_base64)

    async def _on_tts_done(self) -> None:
        """TTS 完成，通知播放器不会再有后续 chunk。"""
        if not self._active or not self._player:
            return
        self._tracer.end(self._turn, "tts")
        self._player.mark_tts_done()

    def _on_playback_done(self) -> None:
        """播放器播完，标记该轮回答完成并回调上层状态机。"""
        if not self._active:
            return

        self._tracer.end(self._turn, "player")

        total = _ms_since(self._t0)
        log.info(f"⏱  Turn complete    +{total}ms total")

        if self._on_text and self._assistant_text:
            asyncio.create_task(self._on_text(self._assistant_text, True))

        finished_tts = self._tts
        self._active = False
        self._tts = None
        self._player = None

        if finished_tts:
            asyncio.create_task(finished_tts.cancel())

        self._on_done()
