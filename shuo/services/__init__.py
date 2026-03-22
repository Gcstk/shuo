"""Lazy exports for external service adapters."""

from importlib import import_module

__all__ = [
    "FluxService",
    "LLMService",
    "TTSService",
    "TTSPool",
    "AudioPlayer",
    "make_outbound_call",
]

_MODULE_MAP = {
    "FluxService": ".flux",
    "LLMService": ".llm",
    "TTSService": ".tts",
    "TTSPool": ".tts_pool",
    "AudioPlayer": ".player",
    "make_outbound_call": ".twilio_client",
}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(name)
    module = import_module(_MODULE_MAP[name], __name__)
    return getattr(module, name)
