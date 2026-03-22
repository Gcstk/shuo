#!/usr/bin/env python3
"""
shuo - Voice Agent Framework

Usage:
    python main.py                  # browser/server mode
    python main.py +1234567890      # outbound call mode

Server-only mode starts the server and serves the browser demo.
Outbound mode additionally initiates a call to the specified number.

启动时序（教学版）：
1. 读取 .env 并校验外部服务密钥
2. 启动 FastAPI（负责 Twilio 回调与 WebSocket）
3. 可选发起 Twilio 外呼
4. 主线程驻留，等待通话生命周期结束

进程结束时，SIGTERM 会触发优雅排空：
- 拒绝新通话
- 等待活跃通话自然结束（最多 DRAIN_TIMEOUT）
- 再退出服务
"""

import os
import sys
import signal
import threading
import time

import uvicorn
from dotenv import load_dotenv

from shuo.server import app
from shuo.services.twilio_client import make_outbound_call
from shuo.log import setup_logging, Logger, get_logger
import shuo.server as server_module

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger("shuo")


def check_environment(require_twilio: bool = False) -> bool:
    """
    校验运行所需环境变量。

    注意：
    - 浏览器/纯服务模式不强依赖 Twilio
    - 只有显式外呼时才校验 Twilio 相关变量
    """
    turn_provider = os.getenv("TURN_PROVIDER", "flux").strip().lower()
    required_vars = []
    tts_provider = os.getenv("TTS_PROVIDER", "elevenlabs").strip().lower()

    if turn_provider in {"flux", "deepgram"}:
        required_vars.append("DEEPGRAM_API_KEY")
    elif turn_provider in {"duplug", "soulx", "soulx-duplug"}:
        required_vars.append("DUPLUG_WS_URL")
    else:
        logger.error(f"Unsupported TURN_PROVIDER: {turn_provider}")
        return False

    if tts_provider == "qwen":
        required_vars.append("DASHSCOPE_API_KEY")
    else:
        required_vars.append("ELEVENLABS_API_KEY")

    if require_twilio:
        required_vars.extend([
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_PHONE_NUMBER",
            "TWILIO_PUBLIC_URL",
        ])
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        return False

    llm_env_vars = [
        "LLM_API_KEY",
        "VLLM_API_KEY",
        "LLM_BASE_URL",
        "VLLM_BASE_URL",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
    ]
    if not any(os.getenv(var) for var in llm_env_vars):
        logger.warning(
            "No explicit LLM environment configured; LLMService will use its internal defaults"
        )
    
    return True


# Max time (seconds) to wait for active calls to finish before forced exit.
DRAIN_TIMEOUT = int(os.getenv("DRAIN_TIMEOUT", "300"))  # 5 minutes default

_uvicorn_server: uvicorn.Server = None


def start_server(port: int) -> None:
    """启动 FastAPI 服务（在后台线程运行）。"""
    global _uvicorn_server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",  # Quiet uvicorn, we have our own logging
    )
    _uvicorn_server = uvicorn.Server(config)
    _uvicorn_server.run()


def main():
    """程序入口：启动服务并可选发起外呼。"""
    phone_number = None

    if len(sys.argv) >= 2:
        phone_number = sys.argv[1]
        if not phone_number.startswith("+"):
            print("Error: Phone number must start with +")
            sys.exit(1)

    # 先校验外部依赖配置，避免半启动状态。
    if not check_environment(require_twilio=bool(phone_number)):
        sys.exit(1)
    
    # Get port from environment
    port = int(os.getenv("PORT", "3040"))
    public_url = os.getenv("TWILIO_PUBLIC_URL", "")
    ready_url = public_url or f"http://localhost:{port}/web"
    
    # 服务线程与主线程分离：
    # - 服务线程跑 uvicorn
    # - 主线程保留给信号处理与外呼控制
    Logger.server_starting(port)
    server_thread = threading.Thread(
        target=start_server,
        args=(port,),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    Logger.server_ready(ready_url)
    
    # ── SIGTERM 优雅下线 ────────────────────────────────────────────
    def _handle_sigterm(signum, frame):
        """
        Railway (and Docker) send SIGTERM before killing the container.
        We stop accepting new calls and wait for active ones to finish.
        """
        logger.info("SIGTERM received — starting graceful drain")
        server_module._draining = True

        # If no active calls, exit immediately
        if server_module._active_calls <= 0:
            logger.info("No active calls — shutting down now")
            if _uvicorn_server:
                _uvicorn_server.should_exit = True
            return

        logger.info(
            f"Waiting up to {DRAIN_TIMEOUT}s for {server_module._active_calls} "
            f"active call(s) to finish..."
        )

        # 轮询等待活跃通话排空，超时则强制退出。
        deadline = time.monotonic() + DRAIN_TIMEOUT
        while server_module._active_calls > 0 and time.monotonic() < deadline:
            time.sleep(1)

        remaining = server_module._active_calls
        if remaining > 0:
            logger.warning(f"Drain timeout — {remaining} call(s) still active, forcing exit")
        else:
            logger.info("All calls drained — shutting down cleanly")

        if _uvicorn_server:
            _uvicorn_server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        if phone_number:
            # 外呼模式：服务起来后，主动让 Twilio 发起拨号。
            Logger.call_initiating(phone_number)
            call_sid = make_outbound_call(phone_number)
            Logger.call_initiated(call_sid)
            logger.info("Waiting for call to connect... (Ctrl+C to end)")
        else:
            # 浏览器/纯服务模式：暴露网页入口，不主动外呼。
            logger.info(f"Browser mode — open {ready_url} (Ctrl+C to end)")

        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        Logger.shutdown()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
