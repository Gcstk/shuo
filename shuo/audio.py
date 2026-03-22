"""Audio conversion helpers for browser and phone transports."""

from __future__ import annotations

import audioop
import base64

PCM_WIDTH = 2
PHONE_SAMPLE_RATE = 8000
BROWSER_FLUX_SAMPLE_RATE = 16000

def b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def pcm16_to_mulaw(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Convert mono PCM16 audio to mulaw 8kHz for Deepgram Flux/Twilio."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    converted = pcm16_resample(pcm_bytes, sample_rate, PHONE_SAMPLE_RATE)
    return audioop.lin2ulaw(converted, PCM_WIDTH)


def pcm16_resample(pcm_bytes: bytes, sample_rate: int, target_sample_rate: int) -> bytes:
    """Resample mono PCM16 audio to a target sample rate."""
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate == target_sample_rate:
        return pcm_bytes
    converted, _ = audioop.ratecv(
        pcm_bytes,
        PCM_WIDTH,
        1,
        sample_rate,
        target_sample_rate,
        None,
    )
    return converted


def mulaw_to_pcm16(mulaw_bytes: bytes, sample_rate: int = PHONE_SAMPLE_RATE) -> bytes:
    """Convert mulaw 8kHz audio to mono PCM16 for browser playback."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, PCM_WIDTH)
    if sample_rate != PHONE_SAMPLE_RATE:
        pcm_bytes, _ = audioop.ratecv(
            pcm_bytes,
            PCM_WIDTH,
            1,
            PHONE_SAMPLE_RATE,
            sample_rate,
            None,
        )
    return pcm_bytes
