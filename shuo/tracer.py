"""
Lightweight span tracer for shuo.

Records begin/end spans and point-in-time markers for each agent turn.
Persists as JSON to <project>/trace/<call_id>.json on call end.

Usage:
    tracer = Tracer()
    tracer.begin_turn(1, "Hello, how are you?")
    tracer.begin(1, "llm")
    tracer.mark(1, "llm_first_token")
    tracer.end(1, "llm")
    tracer.save("MZ8a3b1f")  # -> <project>/trace/MZ8a3b1f.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

from .log import get_logger
from .paths import TRACE_DIR

logger = get_logger("shuo.tracer")


@dataclass
class Span:
    """A named time range within a turn."""
    name: str
    start_ms: float
    end_ms: Optional[float] = None


@dataclass
class Marker:
    """A named point-in-time within a turn."""
    name: str
    time_ms: float


@dataclass
class Turn:
    """All trace data for a single agent turn."""
    turn_number: int
    transcript: str = ""
    t0: float = 0.0  # monotonic reference (not serialized)
    spans: List[Span] = field(default_factory=list)
    markers: List[Marker] = field(default_factory=list)
    cancelled: bool = False


class Tracer:
    """
    Records spans and markers for each agent turn.

    All timestamps are stored as milliseconds relative to the turn's t0.
    """

    def __init__(self) -> None:
        self._turns: Dict[int, Turn] = {}
        self._turn_counter = 0

    def begin_turn(self, transcript: str) -> int:
        # 默认仍以“当前时刻”为 turn 起点，兼容旧调用方。
        return self.begin_turn_at(transcript, start_time=time.monotonic())

    def begin_turn_at(self, transcript: str, start_time: float) -> int:
        """以显式时间起一个 turn，便于把 ASR 侧事件回填到同一时间轴。"""
        self._turn_counter += 1
        turn = Turn(
            turn_number=self._turn_counter,
            transcript=transcript,
            t0=start_time,
        )
        self._turns[self._turn_counter] = turn
        return self._turn_counter

    def has_turn(self, turn: int) -> bool:
        return turn in self._turns

    def get_turn_start(self, turn: int) -> Optional[float]:
        t = self._turns.get(turn)
        if not t:
            return None
        return t.t0

    def update_turn_transcript(self, turn: int, transcript: str) -> None:
        # Flux interim/final 到来后，允许把占位 transcript 更新成最终文本。
        t = self._turns.get(turn)
        if not t or not transcript:
            return
        t.transcript = transcript

    def begin(self, turn: int, name: str) -> None:
        """Begin a named span."""
        t = self._turns.get(turn)
        if not t:
            return
        ms = (time.monotonic() - t.t0) * 1000
        t.spans.append(Span(name=name, start_ms=ms))

    def end(self, turn: int, name: str) -> None:
        """End a named span."""
        t = self._turns.get(turn)
        if not t:
            return
        ms = (time.monotonic() - t.t0) * 1000
        # Find the last span with this name that hasn't been ended
        for span in reversed(t.spans):
            if span.name == name and span.end_ms is None:
                span.end_ms = ms
                return

    def mark(self, turn: int, name: str) -> None:
        """Record a point-in-time marker."""
        t = self._turns.get(turn)
        if not t:
            return
        ms = (time.monotonic() - t.t0) * 1000
        t.markers.append(Marker(name=name, time_ms=ms))

    def mark_at(self, turn: int, name: str, when: float) -> None:
        """按给定 monotonic 时间落点，用于回填 Flux/ASR 事件。"""
        t = self._turns.get(turn)
        if not t:
            return
        ms = (when - t.t0) * 1000
        if ms < 0:
            ms = 0
        t.markers.append(Marker(name=name, time_ms=ms))

    def has_marker(self, turn: int, name: str) -> bool:
        t = self._turns.get(turn)
        if not t:
            return False
        return any(marker.name == name for marker in t.markers)

    def cancel_turn(self, turn: int) -> None:
        """Mark turn as cancelled and end all open spans at current time."""
        t = self._turns.get(turn)
        if not t:
            return
        t.cancelled = True
        ms = (time.monotonic() - t.t0) * 1000
        for span in t.spans:
            if span.end_ms is None:
                span.end_ms = ms

    def save(self, call_id: str) -> Optional[Path]:
        """Write trace data to <project>/trace/<call_id>.json."""
        if not self._turns:
            return None

        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{call_id}.json"

        data = {
            "call_id": call_id,
            "turns": [
                {
                    "turn": t.turn_number,
                    "transcript": t.transcript,
                    "cancelled": t.cancelled,
                    "spans": [asdict(s) for s in t.spans],
                    "markers": [asdict(m) for m in t.markers],
                }
                for t in sorted(self._turns.values(), key=lambda x: x.turn_number)
            ],
        }

        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Trace saved to {path}")
        return path
