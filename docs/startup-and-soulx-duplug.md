# shuo 启动与 SoulX-Duplug 接入指南

## 1. 项目怎么启动

这个项目的默认启动方式是浏览器模式：

```bash
cd /Users/geotk/workspace/audio/shuo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

启动成功后，打开：

```text
http://localhost:3040/web
```

服务入口在 [main.py](/Users/geotk/workspace/audio/shuo/main.py)，页面入口在 [shuo/server.py](/Users/geotk/workspace/audio/shuo/shuo/server.py:57) 的 `/web`，浏览器语音流走 `/ws/browser`。

## 2. 最少需要配置什么

浏览器模式至少需要三类能力：

1. Turn service，也就是 ASR + VAD/turn detection
2. LLM
3. TTS

默认 `.env` 可按下面填：

```dotenv
PORT=3040

TURN_PROVIDER=flux
DEEPGRAM_API_KEY=your_deepgram_api_key

LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=http://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name

TTS_PROVIDER=qwen
DASHSCOPE_API_KEY=your_dashscope_api_key
QWEN_TTS_MODEL=qwen3-tts-flash-realtime-2025-11-27
QWEN_TTS_VOICE=Cherry
QWEN_TTS_LANGUAGE=Chinese
QWEN_TTS_MODE=commit
QWEN_TTS_SAMPLE_RATE=24000
```

如果你改用 ElevenLabs，把下面两项补上即可：

```dotenv
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
```

## 3. `soulx-duplug` 作为 ASR + VAD 能正常运行吗

可以，但前提是你的远端 SoulX-Duplug 服务协议和当前适配器一致。

代码里已经显式支持这三种写法：

- `TURN_PROVIDER=duplug`
- `TURN_PROVIDER=soulx`
- `TURN_PROVIDER=soulx-duplug`

对应逻辑在 [main.py](/Users/geotk/workspace/audio/shuo/main.py:60) 和 [shuo/conversation.py](/Users/geotk/workspace/audio/shuo/shuo/conversation.py:34)。

当前适配器是 [shuo/services/duplug.py](/Users/geotk/workspace/audio/shuo/shuo/services/duplug.py)，它假定远端 WebSocket 协议如下。

客户端发给 SoulX-Duplug：

```json
{
  "type": "audio",
  "session_id": "shuo-xxxx",
  "audio": "<base64 float32 pcm>"
}
```

SoulX-Duplug 回给 shuo：

```json
{
  "type": "turn_state",
  "session_id": "shuo-xxxx",
  "state": {
    "state": "blank|idle|nonidle|speak",
    "text": "...",
    "asr_segment": "...",
    "asr_buffer": "..."
  }
}
```

只要你的服务满足这个协议，就可以正常跑。

这里的状态含义是：

- `nonidle`：用户正在说话，shuo 会把 `asr_buffer` 或 `asr_segment` 当作实时字幕
- `speak`：认为一轮用户说话结束，shuo 会触发 Agent 回复
- `idle`：空闲期更新
- `blank`：忽略

## 4. 使用 `soulx-duplug` 时需要改哪些配置

最小改动只有这两项：

```dotenv
TURN_PROVIDER=soulx-duplug
DUPLUG_WS_URL=ws://your-duplug-host:8000/turn
```

可选项：

```dotenv
DUPLUG_WS_TIMEOUT=10
```

也就是说，和默认 `flux` 模式相比，你只需要：

1. 把 `TURN_PROVIDER` 从 `flux` 改成 `soulx-duplug`
2. 配置 `DUPLUG_WS_URL`
3. 不再要求 `DEEPGRAM_API_KEY`

环境校验逻辑在 [main.py](/Users/geotk/workspace/audio/shuo/main.py:51)。当 `TURN_PROVIDER` 是 `duplug`、`soulx` 或 `soulx-duplug` 时，程序只检查 `DUPLUG_WS_URL`，不会再检查 `DEEPGRAM_API_KEY`。

## 5. 推荐的 `.env` 示例

如果你准备用 SoulX-Duplug + Qwen TTS，可以直接参考这一版：

```dotenv
PORT=3040
DRAIN_TIMEOUT=300

TURN_PROVIDER=soulx-duplug
DUPLUG_WS_URL=ws://your-duplug-host:8000/turn
DUPLUG_WS_TIMEOUT=10

LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=http://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name

TTS_PROVIDER=qwen
DASHSCOPE_API_KEY=your_dashscope_api_key
QWEN_TTS_MODEL=qwen3-tts-flash-realtime-2025-11-27
QWEN_TTS_VOICE=Cherry
QWEN_TTS_LANGUAGE=Chinese
QWEN_TTS_MODE=commit
QWEN_TTS_SAMPLE_RATE=24000
```

## 6. 一条完整的启动路径

```bash
cd /Users/geotk/workspace/audio/shuo
cp .env.example .env
```

编辑 `.env`，改成：

```dotenv
TURN_PROVIDER=soulx-duplug
DUPLUG_WS_URL=ws://127.0.0.1:8000/turn
```

然后启动：

```bash
source .venv/bin/activate
python main.py
```

浏览器打开：

```text
http://localhost:3040/web
```

看到下面这类日志，说明主流程通了：

```text
🚀 Server starting on port 3040
✓  Ready http://localhost:3040/web
🔌 WebSocket connected
▶  Stream started SID: browser-...
```

## 7. 如果跑不起来，优先检查什么

1. `DUPLUG_WS_URL` 是否可连通，是否真的是 `/turn`
2. SoulX-Duplug 返回的消息 `type` 是否为 `turn_state`
3. `state.state` 是否使用了 `blank|idle|nonidle|speak` 这套状态名
4. 远端收到的音频是否接受 `base64(float32 pcm, 16kHz mono)` 这一格式
5. `.env` 是否同时配好了 LLM 和 TTS 所需密钥

最常见的不兼容点不是 shuo 本身，而是远端 SoulX-Duplug 协议字段名或音频编码格式不一致。

## 8. 浏览器启动时，前端是用什么方式写的

不是 React、Vue、Next.js，也不是 Vite 打包产物。

当前前端是：

- FastAPI 直接提供静态文件
- HTML 页面： [shuo/static/browser_agent.html](/Users/geotk/workspace/audio/shuo/shuo/static/browser_agent.html)
- CSS 样式： [shuo/static/browser_agent.css](/Users/geotk/workspace/audio/shuo/shuo/static/browser_agent.css)
- 原生 ES Module JavaScript： [shuo/static/browser_agent.js](/Users/geotk/workspace/audio/shuo/shuo/static/browser_agent.js)
- 麦克风采集：`AudioContext` + `AudioWorklet`
- 实时通信：浏览器原生 `WebSocket`
- 音频播放：Web Audio API

后端通过 [shuo/server.py](/Users/geotk/workspace/audio/shuo/shuo/server.py:57) 的 `/web` 返回 HTML，再挂载 `/static` 目录提供前端资源。

这意味着：

- 前端非常轻，没有构建步骤
- 启动 `python main.py` 就能直接访问页面
- 如果你要改页面，直接改静态文件即可

## 9. 结论

如果你的 SoulX-Duplug 服务协议和 [shuo/services/duplug.py](/Users/geotk/workspace/audio/shuo/shuo/services/duplug.py) 的约定一致，那么 `soulx-duplug` 作为 ASR + VAD 是可以正常运行的。

对这个项目本身来说，接入它只需要改 `TURN_PROVIDER` 和 `DUPLUG_WS_URL`，不需要再改 Python 代码。
