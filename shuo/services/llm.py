"""
LLM 流式服务（当前走 Groq OpenAI-compatible 接口）。

职责：
1. 管理多轮对话历史
2. 发起流式 chat completion
3. 把 token 增量通过回调实时交给上游（Agent -> TTS）
"""

import os
import asyncio
from typing import Optional, Callable, Awaitable, List, Dict

from openai import AsyncOpenAI

from ..log import ServiceLogger

log = ServiceLogger("LLM")

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses concise and conversational, as they will be spoken aloud. Avoid using markdown, bullet points, or other formatting that doesn't work well in speech. Be friendly and natural."""


class LLMService:
    """
    OpenAI-compatible 流式 LLM 服务。
    
    说明：
    - history 在本类内部保存（跨轮上下文）
    - token 通过 on_token 回调逐步上送，而不是一次性返回整段
    """
    
    def __init__(
        self,
        on_token: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ):
        self._on_token = on_token
        self._on_done = on_done

        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("VLLM_API_KEY")
            or os.getenv("GROQ_API_KEY", "sk-geotk")
        )
        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("VLLM_BASE_URL")
            # or "http://192.168.129.25:8000/v1"
            or "http://115.190.220.21:80/v1"
        )
        # self._model = os.getenv("LLM_MODEL", "qwen3.5-4b")
        self._model = os.getenv("LLM_MODEL", "qwen-4b-character")

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
        self._history: List[Dict[str, str]] = []
    
    @property
    def is_active(self) -> bool:
        return self._running and self._task is not None
    
    @property
    def history(self) -> List[Dict[str, str]]:
        return self._history.copy()
    
    def clear_history(self) -> None:
        self._history = []
    
    async def start(self, user_message: str) -> None:
        """启动一轮生成，把用户输入先写入历史。"""
        if self._running:
            await self.cancel()
        
        self._history.append({"role": "user", "content": user_message})
        
        self._running = True
        self._task = asyncio.create_task(self._generate())
        log.connected()
    
    async def cancel(self) -> None:
        """取消当前生成任务。"""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        log.cancelled()
    
    async def _generate(self) -> None:
        """
        执行流式生成并逐 token 回调。

        流式链路关键点：
        - 每收到一个 token 立即回调 on_token
        - Agent 接到 token 后立刻喂给 TTS，形成“边想边说”
        """
        assistant_response = ""
        
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + self._history
            
            # stream=True 开启增量输出，是低延迟语音体验的关键。
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7,
                top_p= 0.95,
                stream_options= {"include_usage": True},
                extra_body={"chat_template_kwargs":{"enable_thinking": False}}
            )
            
            async for chunk in stream:
                if not self._running:
                    break
                
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    token = delta.content
                    assistant_response += token
                    await self._on_token(token)
            
            if self._running and assistant_response:
                self._history.append({"role": "assistant", "content": assistant_response})
                await self._on_done()
        
        except asyncio.CancelledError:
            if assistant_response:
                self._history.append({"role": "assistant", "content": assistant_response + "..."})
            raise
        
        except Exception as e:
            log.error("Generation failed", e)
            await self._on_done()
        
        finally:
            self._running = False
            self._task = None
