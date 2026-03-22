"""
FastAPI server for shuo.

Endpoints:
- GET /health - Health check
- GET/POST /twiml - Returns TwiML for Twilio to connect WebSocket
- WebSocket /ws - Media stream endpoint
- GET /trace/latest - Returns the most recent call trace as JSON
- GET /bench/ttft - Benchmark TTFT across OpenAI models

在主链路里，server.py 主要做两件事：
1. 暴露给 Twilio 的控制面接口（/twiml）与媒体面入口（/ws）
2. 承担通话生命周期的“门面层”职责（健康检查、外呼触发等）

注意：
- /bench/ttft 是观测工具接口，不参与实时通话主链路。
"""

import json
import os
import time
import asyncio
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, Response, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from .conversation import run_conversation_over_twilio, run_conversation_with_transport
from .services.twilio_client import make_outbound_call
from .log import Logger, get_logger
from .transports import BrowserTransport

logger = get_logger("shuo.server")

app = FastAPI(title="shuo", docs_url=None, redoc_url=None)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Graceful shutdown / connection draining ───────────────────────────
_draining = False          # Set True on SIGTERM — reject new calls
_active_calls = 0          # Count of live WebSocket conversations
_drain_event = asyncio.Event()  # Signalled when _active_calls hits 0


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/web")
async def browser_demo():
    """Serve the browser mic/speaker demo."""
    return FileResponse(STATIC_DIR / "browser_agent.html")


@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml():
    """
    返回 TwiML，告诉 Twilio “下一步去连哪个 WebSocket”。
    
    Twilio calls this URL when the call is answered.
    During graceful shutdown, rejects new calls so they don't get cut off.
    """
    if _draining:
        # Reject new calls during shutdown — Twilio will play a message and hang up
        logger.info("Draining — rejecting new inbound call")
        reject_twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, we are updating. Please call back in a moment.</Say>
    <Hangup/>
</Response>"""
        return Response(content=reject_twiml, media_type="application/xml")

    # Twilio 只能访问公网地址，所以本地开发通常用 ngrok 暴露。
    public_url = os.getenv("TWILIO_PUBLIC_URL", "")
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/ws"
    
    # track=inbound_track：只接收来电方音频上行到本服务。
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect record="record-from-answer-dual">
        <Stream url="{ws_url}" track="inbound_track" />
    </Connect>
</Response>"""
    
    return Response(content=twiml_response, media_type="application/xml")


@app.get("/trace/latest")
async def latest_trace():
    """返回最近一通电话的 trace（用于排查延迟/中断）。"""
    trace_dir = Path("/tmp/shuo")
    if not trace_dir.exists():
        return JSONResponse({"error": "No traces found"}, status_code=404)

    traces = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        return JSONResponse({"error": "No traces found"}, status_code=404)

    data = json.loads(traces[0].read_text())
    return JSONResponse(data)


@app.get("/call/{phone_number:path}")
async def trigger_call(phone_number: str):
    """
    触发外呼（调试/演示入口）。

    Usage:
        curl https://your-server/call/+1234567890
    """
    if not phone_number.startswith("+"):
        phone_number = f"+{phone_number}"
    try:
        call_sid = make_outbound_call(phone_number)
        return {"status": "calling", "to": phone_number, "call_sid": call_sid}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


## ── TTFT Benchmark ──────────────────────────────────────────────

BENCH_PROMPT = "Explain how a combustion engine works."

# Each entry: (display_name, provider_key, model_id)
# provider_key is used to look up the right AsyncOpenAI client
DEFAULT_MODELS = [
    # OpenAI 4-series
    ("gpt-4o-mini",   "openai", "gpt-4o-mini"),
    ("gpt-4o",        "openai", "gpt-4o"),
    ("gpt-4.1-nano",  "openai", "gpt-4.1-nano"),
    ("gpt-4.1-mini",  "openai", "gpt-4.1-mini"),
    ("gpt-4.1",       "openai", "gpt-4.1"),
    # OpenAI 5-series
    ("gpt-5-nano",    "openai", "gpt-5-nano"),
    ("gpt-5-mini",    "openai", "gpt-5-mini"),
    ("gpt-5",         "openai", "gpt-5"),
    ("gpt-5.1",       "openai", "gpt-5.1"),
    ("gpt-5.2",       "openai", "gpt-5.2"),
    # Groq
    ("groq/llama-3.3-70b",  "groq", "llama-3.3-70b-versatile"),
    ("groq/llama-3.1-8b",   "groq", "llama-3.1-8b-instant"),
]

BENCH_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": BENCH_PROMPT},
]


def _make_clients() -> dict:
    """
    构建 provider -> OpenAI 兼容客户端映射。

    说明：
    - openai 使用官方端点
    - groq 通过 OpenAI-compatible base_url 访问
    """
    clients = {}
    oai_key = os.getenv("OPENAI_API_KEY", "")
    if oai_key:
        clients["openai"] = AsyncOpenAI(api_key=oai_key)
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        clients["groq"] = AsyncOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return clients


async def _measure_ttft(client: AsyncOpenAI, model: str) -> float:
    """
    Single TTFT measurement in milliseconds.

    打开流式 completion，记录首个内容 token 到达时间后立即关闭。
    """
    # GPT-5+ uses max_completion_tokens; older models use max_tokens
    is_new = model.startswith(("gpt-5", "o1", "o3", "o4"))
    token_param = "max_completion_tokens" if is_new else "max_tokens"

    params: dict = {
        "model": model,
        "messages": BENCH_MESSAGES,
        "stream": True,
        token_param: 20,
    }
    if is_new:
        # Use lowest reasoning effort the model accepts:
        # try "none" first, fall back to "minimal"
        params["extra_body"] = {"reasoning_effort": "none"}
    else:
        params["temperature"] = 0

    t0 = time.perf_counter()
    try:
        stream = await client.chat.completions.create(**params)
    except Exception as e:
        if is_new and "none" in str(e).lower():
            # Model doesn't support "none" — retry with "minimal"
            params["extra_body"] = {"reasoning_effort": "minimal"}
            t0 = time.perf_counter()
            stream = await client.chat.completions.create(**params)
        else:
            raise
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            ttft_ms = (time.perf_counter() - t0) * 1000
            await stream.close()
            return ttft_ms
    # edge case: no content tokens at all
    return (time.perf_counter() - t0) * 1000



@app.get("/bench/ttft")
async def bench_ttft(
    models: Optional[str] = Query(
        None,
        description="Comma-separated model names. Defaults to a built-in list.",
    ),
    runs: int = Query(30, ge=1, le=100, description="Runs per model"),
):
    """
    Benchmark TTFT across OpenAI-compatible models.

    Usage:
        curl https://your-server/bench/ttft
        curl https://your-server/bench/ttft?models=gpt-4o-mini,gpt-4o&runs=5
    """
    clients = _make_clients()

    # Build model list: use DEFAULT_MODELS or parse comma-separated overrides
    if models:
        # For custom input, assume openai provider unless "groq/" prefixed
        entries = []
        for m in models.split(","):
            m = m.strip()
            if not m:
                continue
            if m.startswith("groq/"):
                entries.append((m, "groq", m.removeprefix("groq/")))
            else:
                entries.append((m, "openai", m))
        model_entries = entries
    else:
        model_entries = DEFAULT_MODELS

    # Filter out models whose provider has no API key
    model_entries = [(name, prov, mid) for name, prov, mid in model_entries if prov in clients]

    # Build a shuffled schedule: each model appears `runs` times, interleaved
    schedule = [(name, prov, mid, i) for name, prov, mid in model_entries for i in range(runs)]
    random.shuffle(schedule)

    total = len(schedule)
    names = [name for name, _, _ in model_entries]
    logger.info(f"TTFT benchmark: {len(model_entries)} models × {runs} runs = {total} calls (randomised)")

    times_by_model: dict[str, list[float]] = defaultdict(list)
    errors_by_model: dict[str, list[str]] = defaultdict(list)

    for idx, (name, prov, mid, run_i) in enumerate(schedule, 1):
        try:
            ms = await _measure_ttft(clients[prov], mid)
            times_by_model[name].append(round(ms, 1))
            logger.info(f"  [{idx}/{total}] {name} #{run_i+1} → {ms:.0f} ms")
        except Exception as e:
            errors_by_model[name].append(f"run {run_i+1}: {e}")
            logger.info(f"  [{idx}/{total}] {name} #{run_i+1} → ERROR")

    # Aggregate stats per model (preserve original order)
    results = []
    for name in names:
        t = times_by_model.get(name, [])
        errs = errors_by_model.get(name, [])
        if not t:
            results.append({"model": name, "error": errs[0] if errs else "no data"})
            logger.info(f"  {name} → ERROR: {errs[0] if errs else 'no data'}")
            continue
        avg = round(sum(t) / len(t), 1)
        entry: dict = {
            "model": name,
            "runs": len(t),
            "avg_ms": avg,
            "min_ms": min(t),
            "max_ms": max(t),
            "all_ms": t,
        }
        if errs:
            entry["errors"] = errs
        results.append(entry)
        logger.info(f"  {name} → avg {avg} ms  (min {min(t)}, max {max(t)})")

    return JSONResponse({
        "prompt": BENCH_PROMPT,
        "runs_per_model": runs,
        "results": results,
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Twilio Media Streams 的双向 WebSocket 入口。
    
    这里承接实时双工音频：
    - 上行：Twilio -> conversation -> Flux
    - 下行：Agent(TTS/Player) -> Twilio

    同时维护 active_calls，供 SIGTERM 优雅排空使用。
    """
    global _active_calls

    # Twilio 连接建立后，一通电话对应一个 run_conversation_over_twilio。
    await websocket.accept()
    _active_calls += 1
    logger.info(f"Call connected  (active: {_active_calls})")

    try:
        await run_conversation_over_twilio(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        _active_calls -= 1
        logger.info(f"Call ended  (active: {_active_calls})")
        if _draining and _active_calls <= 0:
            _drain_event.set()


@app.websocket("/ws/browser")
async def browser_websocket_endpoint(websocket: WebSocket):
    """Browser realtime voice endpoint."""
    global _active_calls

    await websocket.accept()
    Logger.websocket_connected()
    _active_calls += 1
    logger.info(f"Browser session connected  (active: {_active_calls})")

    transport = BrowserTransport(websocket)
    await transport.send_ready()

    try:
        await run_conversation_with_transport(transport)
    except Exception as e:
        logger.error(f"Browser WebSocket error: {e}")
    finally:
        _active_calls -= 1
        logger.info(f"Browser session ended  (active: {_active_calls})")
        if _draining and _active_calls <= 0:
            _drain_event.set()
