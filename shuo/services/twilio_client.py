"""
Twilio 适配层：
1. 通过 REST API 发起外呼
2. 解析 Twilio Media Streams WebSocket 消息

Twilio 在这里承担两类职责：
- 信令/控制：calls.create 发起电话
- 媒体传输：start/media/stop 消息流
"""

import os
import json
import base64
from typing import Optional

from ..types import (
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
)
from ..log import Logger


def make_outbound_call(to_number: str) -> str:
    """
    使用 Twilio 发起外呼。
    
    Args:
        to_number: Phone number to call in E.164 format (+1234567890)
        
    Returns:
        Call SID
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    public_url = os.getenv("TWILIO_PUBLIC_URL")
    
    if not all([account_sid, auth_token, from_number, public_url]):
        raise ValueError("Missing required Twilio environment variables")
    
    # 指定 edge/region 以控制网络路径与稳定性。
    from twilio.rest import Client

    client = Client(account_sid, auth_token, edge="frankfurt", region="us1")
    
    # 呼叫接通后，Twilio 会请求该 URL 获取下一步 TwiML 指令。
    twiml_url = f"{public_url}/twiml"
    
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=twiml_url,
        record=True,
    )
    
    return call.sid


def parse_twilio_message(data: dict) -> Optional[Event]:
    """
    把 Twilio 原始 WS 消息解析为内部事件。

    Twilio 典型事件：
    - connected: WebSocket 已连接（仅用于日志）
    - start:     媒体流开始，包含 streamSid
    - media:     base64 编码音频 payload（mulaw 8k）
    - stop:      媒体流结束
    """
    event_type = data.get("event")

    if event_type == "connected":
        Logger.websocket_connected()
        return None

    elif event_type == "start":
        start_data = data.get("start", {})
        stream_sid = start_data.get("streamSid")
        if stream_sid:
            return StreamStartEvent(stream_sid=stream_sid)

    elif event_type == "media":
        media_data = data.get("media", {})
        payload = media_data.get("payload", "")
        if payload:
            # Twilio 发送的是 base64 字符串，先解码为 bytes 再喂给 Flux。
            audio_bytes = base64.b64decode(payload)
            return MediaEvent(audio_bytes=audio_bytes)

    elif event_type == "stop":
        return StreamStopEvent()

    return None
