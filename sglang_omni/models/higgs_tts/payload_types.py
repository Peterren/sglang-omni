# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for Higgs TTS.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class HiggsTtsState(DeclarativeStateBase):
    """Per-request state threaded through preprocessing -> audio_encoder ->
    tts_engine -> vocoder. Fields populate lazily so a deserialised state is
    valid at any stage boundary."""

    # note (luojiaxuan): Emit sample_rate only alongside audio_samples because
    # the rate is meaningless until the vocoder has produced a waveform.
    sample_rate: int = wire(24000, emit="with:audio_samples")

    # preprocessing / audio_encoder
    prompt_token_ids: list[int] = wire(default_factory=list, codec="list")
    reference_codes_delayed: list[list[int]] | None = None
    target_text: str | None = None
    reference_text: str | None = None
    reference_waveform: Any | None = None  # mono 24 kHz [1, 1, L] torch.Tensor
    reference_code_cache_key: str | None = None
    uploaded_voice_name: str | None = None
    uploaded_voice_created_at: int | None = None

    num_codebooks: int = 8
    codebook_size: int = 1026  # 1024 data + <|boc|> + <|eoc|>

    # generation params
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    # RL rollout controls
    return_logprob: bool = wire(False, emit="truthy", codec="bool")
    return_omni_rollout: bool = wire(False, emit="truthy", codec="bool")

    # tts_engine
    output_codes_delayed: list[list[int]] | None = None
    omni_rollout: dict[str, Any] | None = None

    # vocoder
    audio_samples: Any | None = None


__all__ = ["HiggsTtsState"]
