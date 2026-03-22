"""
shuo 的纯状态机。

核心函数:
    process_event(state, event) -> (new_state, actions)

这层只做“决策”，不做网络 I/O。
可以把它理解成一个路由器：
1. 输入事件（来自 Twilio/Flux/Agent）
2. 输出动作（喂给 Flux / 启动回答 / 打断回答）

轮次检测由 Deepgram Flux 完成，因此本地状态机逻辑保持很小。
"""

from dataclasses import replace
from typing import List, Tuple

from .types import (
    AppState, Phase,
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
    FluxStartOfTurnEvent, FluxEndOfTurnEvent, AgentTurnDoneEvent,
    Action, FeedFluxAction, StartAgentTurnAction, ResetAgentTurnAction,
)


def process_event(state: AppState, event: Event) -> Tuple[AppState, List[Action]]:
    """
    纯函数状态机: (State, Event) -> (State, Actions)

    事件到动作的映射规则：
    - MediaEvent          -> 持续喂音频到 Flux（流式识别不断流）
    - FluxEndOfTurnEvent  -> 结束一轮用户发言，启动 Agent 回答
    - FluxStartOfTurnEvent-> 用户抢话，打断 Agent
    - AgentTurnDoneEvent  -> Agent 播报完成，回到 LISTENING
    """
    # Twilio 流建立后进入 LISTENING，等待用户说话。
    if isinstance(event, StreamStartEvent):
        return replace(state, stream_sid=event.stream_sid, phase=Phase.LISTENING), []

    # Twilio 流结束时，如果还在回答，需要先发出重置动作做善后。
    if isinstance(event, StreamStopEvent):
        actions: List[Action] = []
        if state.phase == Phase.RESPONDING:
            actions.append(ResetAgentTurnAction())
        return state, actions

    # 关键点：无论当前 phase 是什么，音频都持续送入 Flux。
    # 这保证了流式 ASR 与 turn detection 始终在线。
    if isinstance(event, MediaEvent):
        return state, [FeedFluxAction(audio_bytes=event.audio_bytes)]

    # 只有在 LISTENING 时，才用 EndOfTurn 启动回答。
    if isinstance(event, FluxEndOfTurnEvent):
        if event.transcript and state.phase == Phase.LISTENING:
            new_state = replace(state, phase=Phase.RESPONDING)
            return new_state, [StartAgentTurnAction(transcript=event.transcript)]
        # 如果已经在 RESPONDING，说明是旧事件或重复事件，忽略更安全。
        return state, []

    # StartOfTurn 发生在 RESPONDING 时，就是“抢话打断”信号。
    # 状态切回 LISTENING，并触发 Reset 取消当前播报链路。
    if isinstance(event, FluxStartOfTurnEvent):
        if state.phase == Phase.RESPONDING:
            return replace(state, phase=Phase.LISTENING), [ResetAgentTurnAction()]
        return state, []

    # Agent 播报结束，回到 LISTENING，等待下一轮用户输入。
    if isinstance(event, AgentTurnDoneEvent):
        if state.phase == Phase.RESPONDING:
            return replace(state, phase=Phase.LISTENING), []
        return state, []

    return state, []
