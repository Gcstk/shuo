"""
shuo 的类型定义。

阅读本文件时可以把它当成“系统词汇表”：
1. State: 当前系统处于什么阶段（只保留路由决策所需的最小信息）
2. Event: 外部输入/内部回调触发的事件
3. Action: 状态机给出的副作用指令（真正 I/O 在其他模块执行）

这里有一个关键设计：AppState 故意保持很轻，不存会话历史。
会话历史在 Agent/LLM 内部维护，这样状态机仍然是纯函数、可测试。
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union, List


# =============================================================================
# STATE
# =============================================================================

class Phase(Enum):
    """当前对话阶段。"""
    LISTENING = auto()    # Waiting for user / user speaking
    RESPONDING = auto()   # Agent active (LLM -> TTS -> Playback)


@dataclass(frozen=True)
class AppState:
    """
    应用状态：只保存路由所需信息。

    说明：
    - phase 决定下一步如何处理事件（继续听/正在回答）
    - stream_sid 用于和 Twilio 当前流对应
    - 不保存历史对话，历史由 Agent 持有
    """
    phase: Phase = Phase.LISTENING
    stream_sid: Optional[str] = None


# =============================================================================
# EVENTS (inputs to the system)
# =============================================================================

@dataclass(frozen=True)
class StreamStartEvent:
    """Twilio 媒体流开始事件。"""
    stream_sid: str


@dataclass(frozen=True)
class StreamStopEvent:
    """Twilio 媒体流结束事件。"""
    pass


@dataclass(frozen=True)
class MediaEvent:
    """从 Twilio 收到的音频帧。"""
    audio_bytes: bytes


@dataclass(frozen=True)
class FluxStartOfTurnEvent:
    """
    Deepgram Flux 检测到用户开始说话。

    典型用途是打断（barge-in）：
    - 当 Agent 正在播报时，用户开口
    - 状态机会触发 ResetAgentTurnAction 来取消当前回答并清空播放缓冲
    """
    pass


@dataclass(frozen=True)
class FluxEndOfTurnEvent:
    """
    Deepgram Flux 检测到用户一句话结束。

    会携带该轮识别文本 transcript，状态机会据此触发 Agent 开始回答。
    """
    transcript: str


@dataclass(frozen=True)
class AgentTurnDoneEvent:
    """Agent 播报完成（播放器完成下行音频输出）。"""
    pass


Event = Union[
    StreamStartEvent, StreamStopEvent, MediaEvent,
    FluxStartOfTurnEvent, FluxEndOfTurnEvent,
    AgentTurnDoneEvent,
]


# =============================================================================
# ACTIONS (outputs from the system)
# =============================================================================

@dataclass(frozen=True)
class FeedFluxAction:
    """把 Twilio 上行音频继续喂给 Deepgram Flux。"""
    audio_bytes: bytes


@dataclass(frozen=True)
class StartAgentTurnAction:
    """启动 Agent 一轮回答（LLM -> TTS -> 播放）。"""
    transcript: str


@dataclass(frozen=True)
class ResetAgentTurnAction:
    """
    取消当前回答并清空 Twilio 播放缓冲。

    常见触发场景：用户抢话（StartOfTurn）或流结束时的清理。
    """
    pass


Action = Union[
    FeedFluxAction,
    StartAgentTurnAction,
    ResetAgentTurnAction,
]
