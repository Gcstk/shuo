"""Shared conversation loop for Twilio and browser transports."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import WebSocket

from .agent import Agent
from .log import Logger, get_logger
from .services.flux import FluxService, FluxTurnInfo
from .services.tts_pool import TTSPool
from .state import process_event
from .transports import BaseTransport, TwilioTransport
from .tracer import Tracer
from .types import (
    AgentTurnDoneEvent,
    AppState,
    Event,
    FeedFluxAction,
    FluxEndOfTurnEvent,
    FluxStartOfTurnEvent,
    ResetAgentTurnAction,
    StartAgentTurnAction,
    StreamStartEvent,
    StreamStopEvent,
)

logger = get_logger("shuo.conversation")


async def _noop_text(_text: str, _final: bool) -> None:
    return None


def _build_turn_service(**callbacks):
    provider = os.getenv("TURN_PROVIDER", "flux").strip().lower()
    if provider in {"duplug", "soulx", "soulx-duplug"}:
        from .services.duplug import DuplugService

        logger.info("Using SoulX-Duplug turn service")
        return DuplugService(**callbacks)
    if provider in {"flux", "deepgram"}:
        return FluxService(**callbacks)
    raise ValueError(f"Unsupported TURN_PROVIDER: {provider}")


async def run_conversation_with_transport(transport: BaseTransport) -> None:
    """Run a single bidirectional conversation over an arbitrary transport."""
    event_log = Logger(verbose=False)
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    tracer = Tracer()

    agent: Optional[Agent] = None
    tts_pool_size = max(1, int(os.getenv("TTS_POOL_SIZE", "1")))
    tts_pool_ttl = max(0.1, float(os.getenv("TTS_POOL_TTL", "8.0")))
    tts_pool = TTSPool(pool_size=tts_pool_size, ttl=tts_pool_ttl)
    stream_sid: Optional[str] = None
    # Deepgram 的 turn_index 和本地 tracer turn_id 不是同一个概念，
    # 这里做一次映射，保证 ASR 和下游响应链路落在同一张图上。
    flux_trace_turns: dict[int, int] = {}
    pending_agent_trace_turn: Optional[int] = None

    async def on_flux_end_of_turn(transcript: str) -> None:
        nonlocal pending_agent_trace_turn
        # EndOfTurn 到来时，下一步启动 Agent 的就是这轮用户 turn。
        pending_agent_trace_turn = _latest_trace_turn()
        await event_queue.put(FluxEndOfTurnEvent(transcript=transcript))

    async def on_flux_start_of_turn() -> None:
        await event_queue.put(FluxStartOfTurnEvent())

    async def on_flux_interim(transcript: str) -> None:
        if transport.supports_live_text:
            await transport.send_transcript("user", transcript, False)

    def _latest_trace_turn() -> Optional[int]:
        if not flux_trace_turns:
            return None
        return max(flux_trace_turns.values())

    async def on_flux_turn_info(info: FluxTurnInfo) -> None:
        if info.turn_index is None:
            return

        trace_turn = flux_trace_turns.get(info.turn_index)
        if trace_turn is None:
            # Flux 的 audio_window_end 表示这轮用户音频已经覆盖到哪里。
            # 用“收到事件的本地时间 - window_end”回推近似的用户开口时刻。
            estimated_turn_start = max(
                info.received_at - max(info.audio_window_end, 0.0),
                0.0,
            )
            trace_turn = tracer.begin_turn_at(
                transcript=info.transcript,
                start_time=estimated_turn_start,
            )
            flux_trace_turns[info.turn_index] = trace_turn
            tracer.mark_at(trace_turn, "user_audio_first_frame", estimated_turn_start)

        if info.transcript:
            tracer.update_turn_transcript(trace_turn, info.transcript)

        if info.event == "StartOfTurn" and not tracer.has_marker(trace_turn, "flux_start_of_turn"):
            tracer.mark_at(trace_turn, "flux_start_of_turn", info.received_at)

        if (
            info.transcript
            and info.event in {"StartOfTurn", "Update", "TurnResumed", "EagerEndOfTurn"}
            and not tracer.has_marker(trace_turn, "asr_first_interim")
        ):
            # 首次拿到非空 transcript 时，记为“ASR 首字/首段可见”时间点。
            tracer.mark_at(trace_turn, "asr_first_interim", info.received_at)

        if info.event == "EndOfTurn":
            turn_start = tracer.get_turn_start(trace_turn) or info.received_at
            # audio_window_end 对应这轮用户最后一个被纳入 turn 的音频位置，
            # 用它来估算“用户最后说话时刻”。
            user_last_audio_time = turn_start + max(info.audio_window_end, 0.0)
            if not tracer.has_marker(trace_turn, "user_last_audio_frame"):
                tracer.mark_at(trace_turn, "user_last_audio_frame", user_last_audio_time)
            if not tracer.has_marker(trace_turn, "flux_end_of_turn"):
                tracer.mark_at(trace_turn, "flux_end_of_turn", info.received_at)
            if not tracer.has_marker(trace_turn, "asr_final_transcript"):
                tracer.mark_at(trace_turn, "asr_final_transcript", info.received_at)

    flux = _build_turn_service(
        on_end_of_turn=on_flux_end_of_turn,
        on_start_of_turn=on_flux_start_of_turn,
        on_interim=on_flux_interim if transport.supports_live_text else None,
        on_turn_info=on_flux_turn_info,
    )

    async def read_transport() -> None:
        try:
            while True:
                event = await transport.receive_event()
                if event is None:
                    continue
                await event_queue.put(event)
                if isinstance(event, StreamStopEvent):
                    break
        except Exception as exc:
            event_log.error("Transport reader", exc)
            await event_queue.put(StreamStopEvent())

    state = AppState()
    reader_task = asyncio.create_task(read_transport())

    try:
        while True:
            event = await event_queue.get()
            event_log.event(event)

            if isinstance(event, StreamStartEvent):
                stream_sid = event.stream_sid
                await flux.start(
                    encoding=transport.flux_encoding,
                    sample_rate=transport.flux_sample_rate,
                )
                await tts_pool.start()
                agent = Agent(
                    send_audio=transport.send_audio_chunk,
                    clear_audio=transport.clear_audio,
                    on_done=lambda: event_queue.put_nowait(AgentTurnDoneEvent()),
                    tts_pool=tts_pool,
                    tracer=tracer,
                    on_text=(
                        (lambda text, final: transport.send_transcript("assistant", text, final))
                        if transport.supports_live_text
                        else _noop_text
                    ),
                )

            if transport.supports_live_text and isinstance(event, FluxEndOfTurnEvent):
                await transport.send_transcript("user", event.transcript, True)

            old_phase = state.phase
            state, actions = process_event(state, event)
            event_log.transition(old_phase, state.phase)

            if isinstance(event, StreamStartEvent) or old_phase != state.phase:
                await transport.send_phase(state.phase)

            for action in actions:
                event_log.action(action)
                if isinstance(action, FeedFluxAction):
                    await flux.send(action.audio_bytes)
                elif isinstance(action, StartAgentTurnAction):
                    if agent:
                        await agent.start_turn(
                            action.transcript,
                            trace_turn=pending_agent_trace_turn,
                        )
                        pending_agent_trace_turn = None
                elif isinstance(action, ResetAgentTurnAction):
                    if agent:
                        await agent.cancel_turn()

            if isinstance(event, StreamStopEvent):
                break

    except Exception as exc:
        event_log.error("Call loop", exc)
        raise

    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass

        if agent:
            await agent.cleanup()

        await tts_pool.stop()
        await flux.stop()
        await transport.close()

        call_id = stream_sid or "unknown"
        tracer.save(call_id)
        Logger.websocket_disconnected()


async def run_conversation_over_twilio(websocket: WebSocket) -> None:
    """Backward-compatible entrypoint for Twilio media streams."""
    await run_conversation_with_transport(TwilioTransport(websocket))
