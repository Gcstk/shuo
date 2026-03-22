"""
ElevenLabs TTS connection pool.

Pre-connects WebSocket connections and manages their lifecycle.
Stale connections (past TTL) are evicted automatically.
The pool auto-refills after a connection is dispensed.

Usage:
    pool = TTSPool(pool_size=1, ttl=8.0)
    await pool.start()

    tts = await pool.get(on_audio=..., on_done=...)
    # tts is ready to use immediately (if warm) or after a fresh connect

    await pool.stop()

为什么需要池化：
1. TTS 建连成本通常高于发一次文本 chunk
2. 若每轮都临时建连，首音频延迟会明显上升
3. 预热连接可显著降低“开始说话”等待时间
"""

import asyncio
import time
from typing import Optional, Callable, Awaitable, List
from dataclasses import dataclass

from .tts import TTSService
from ..log import ServiceLogger

log = ServiceLogger("TTSPool")


# No-op callbacks for pre-connected (idle) services
async def _noop_audio(_audio: str) -> None:
    pass


async def _noop_done() -> None:
    pass


@dataclass
class _Entry:
    """A pooled TTS connection with its creation timestamp."""
    tts: TTSService
    created_at: float  # time.monotonic()


class TTSPool:
    """
    ElevenLabs TTS WebSocket 连接池。

    行为：
    - 启动后预热 pool_size 条连接
    - get() 优先发放 warm 连接（并重绑回调）
    - 超过 ttl 的空闲连接会淘汰
    - 淘汰/发放后后台自动补齐
    """

    def __init__(self, pool_size: int = 1, ttl: float = 8.0):
        self._pool_size = pool_size
        self._ttl = ttl

        self._ready: List[_Entry] = []
        self._running = False
        self._fill_event = asyncio.Event()
        self._fill_task: Optional[asyncio.Task] = None

    @property
    def available(self) -> int:
        """Number of warm connections ready to dispense."""
        return len(self._ready)

    async def start(self) -> None:
        """启动连接池后台补池任务。"""
        if self._running:
            return

        self._running = True
        self._fill_task = asyncio.create_task(self._fill_loop())

    async def get(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> TTSService:
        """
        获取可用 TTS 连接，并绑定本轮回调。

        优先返回 warm 且未过期连接；
        如果池空或都过期，则阻塞创建新连接。
        """
        # 先尝试发放 warm 连接，减少首音频等待。
        while self._ready:
            entry = self._ready.pop(0)
            age = time.monotonic() - entry.created_at

            if age < self._ttl:
                entry.tts.bind(on_audio, on_done)
                age_ms = int(age * 1000)
                log.info(f"Dispensed warm connection (idle {age_ms}ms)")
                self._trigger_fill()
                return entry.tts
            else:
                age_ms = int(age * 1000)
                log.info(f"Discarded stale connection (idle {age_ms}ms)")
                await entry.tts.cancel()

        # 没有可用 warm 连接时，降级为同步建连。
        log.info("Pool empty, connecting fresh...")
        tts = TTSService(on_audio=on_audio, on_done=on_done)
        await tts.start()
        self._trigger_fill()
        return tts

    async def stop(self) -> None:
        """Shut down pool and clean up all connections."""
        self._running = False
        self._fill_event.set()  # unblock fill loop

        if self._fill_task:
            self._fill_task.cancel()
            try:
                await self._fill_task
            except asyncio.CancelledError:
                pass
            self._fill_task = None

        for entry in self._ready:
            await entry.tts.cancel()
        self._ready.clear()

    def _trigger_fill(self) -> None:
        """Signal the fill loop to check pool levels."""
        self._fill_event.set()

    async def _fill_loop(self) -> None:
        """
        后台补池循环：负责“淘汰旧连接 + 补齐目标容量”。

        即使没有外部触发，也会按 ttl/2 周期做健康刷新。
        """
        try:
            while self._running:
                # Evict stale entries
                await self._evict_stale()

                # Fill to target
                while self._running and len(self._ready) < self._pool_size:
                    tts = TTSService(on_audio=_noop_audio, on_done=_noop_done)
                    try:
                        await tts.start()
                        self._ready.append(
                            _Entry(tts=tts, created_at=time.monotonic())
                        )
                        log.info(
                            f"🔥 Warm connection ready "
                            f"({len(self._ready)}/{self._pool_size})"
                        )
                    except Exception as e:
                        log.error("Pre-connect failed", e)
                        await asyncio.sleep(1.0)  # back off

                # Wait for signal (dispensed/evicted) or periodic refresh
                self._fill_event.clear()
                try:
                    await asyncio.wait_for(
                        self._fill_event.wait(),
                        timeout=self._ttl / 2,
                    )
                except asyncio.TimeoutError:
                    pass  # periodic staleness check

        except asyncio.CancelledError:
            pass

    async def _evict_stale(self) -> None:
        """Remove connections that have been idle past TTL."""
        now = time.monotonic()
        fresh: List[_Entry] = []

        for entry in self._ready:
            age = now - entry.created_at
            if age < self._ttl:
                fresh.append(entry)
            else:
                age_ms = int(age * 1000)
                log.info(f"Evicted stale connection (idle {age_ms}ms)")
                await entry.tts.cancel()

        self._ready = fresh
