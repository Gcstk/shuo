"""
Transport-neutral audio player.

Manages its own independent playback loop that drips audio
chunks at the correct rate, regardless of other activity.

这个模块是下行音频的“节拍器”：
- 上游可以突发地产生很多 chunk
- Player 按约 20ms 间隔稳定发送给下游 transport
- 遇到打断时可立即 clear，清掉 transport 侧缓冲
"""

import asyncio
from typing import Awaitable, Callable, List, Optional

from ..log import ServiceLogger

log = ServiceLogger("Player")


class AudioPlayer:
    """
    按正确节奏把音频流发送到 transport。
    
    特性：
    - 独立播放循环，不阻塞主事件循环
    - 支持动态追加 chunk（适配流式 TTS）
    - 支持即时 stop + clear（抢话打断）
    - 播放完毕回调通知上层
    """
    
    def __init__(
        self,
        send_audio: Callable[[str], Awaitable[None]],
        send_clear: Callable[[], Awaitable[None]],
        on_done: Optional[Callable[[], None]] = None,
    ):
        self._send_audio_cb = send_audio
        self._send_clear_cb = send_clear
        self._on_done = on_done
        
        self._chunks: List[str] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._index = 0
        self._tts_done = False
    
    @property
    def is_playing(self) -> bool:
        return self._running and self._task is not None and not self._task.done()
    
    async def start(self) -> None:
        """Start the playback loop."""
        if self.is_playing:
            await self.stop_and_clear()
        
        self._chunks = []
        self._index = 0
        self._running = True
        self._tts_done = False
        
        self._task = asyncio.create_task(self._playback_loop())
    
    async def send_chunk(self, chunk: str) -> None:
        """追加一段待播放音频。"""
        if not self._running:
            await self.start()
        
        self._chunks.append(chunk)
    
    def mark_tts_done(self) -> None:
        """Signal that TTS is complete - no more chunks coming."""
        self._tts_done = True
    
    async def play(self, chunks: List[str]) -> None:
        """Start playing a fixed list of audio chunks (legacy mode)."""
        if self.is_playing:
            await self.stop_and_clear()
        
        self._chunks = list(chunks)
        self._index = 0
        self._running = True
        self._tts_done = True
        
        self._task = asyncio.create_task(self._playback_loop())
    
    async def stop_and_clear(self) -> None:
        """
        立即停止播放并通知 Twilio 清缓冲。

        这是 barge-in 体验的关键动作：防止旧回复继续播出。
        """
        self._running = False
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self._task = None
        self._chunks = []
        self._index = 0
        self._tts_done = False
        
        await self._send_clear()
    
    async def wait_until_done(self) -> None:
        """Wait for playback to complete (or be interrupted)."""
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _playback_loop(self) -> None:
        """
        独立播放循环，按 ~20ms 节奏 drip 发送。

        为什么要定速：
        - 下游 transport 通常期望稳定媒体节奏
        - 可避免突发发送造成抖动或缓冲异常
        """
        try:
            while self._running:
                if self._index < len(self._chunks):
                    chunk = self._chunks[self._index]
                    await self._send_audio(chunk)
                    self._index += 1
                    await asyncio.sleep(0.020)
                    
                elif self._tts_done:
                    break
                else:
                    await asyncio.sleep(0.010)
            
            if self._running:
                self._running = False
                if self._on_done:
                    self._on_done()
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("Playback failed", e)
            self._running = False
    
    async def _send_audio(self, payload: str) -> None:
        """向下游 transport 发送一个音频 chunk。"""
        await self._send_audio_cb(payload)

    async def _send_clear(self) -> None:
        """发送 clear 事件，让下游 transport 丢弃尚未播放的缓冲音频。"""
        await self._send_clear_cb()
