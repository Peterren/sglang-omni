# SPDX-License-Identifier: Apache-2.0
"""Pre-tokenized Higgs TTS rollout input helpers."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState


def is_pretokenized_prompt(inputs: Any) -> bool:
    """Return true for the ``/generate input_ids`` rollout shape."""
    return (
        isinstance(inputs, list)
        and bool(inputs)
        and all(isinstance(token, int) for token in inputs)
    )


def build_pretokenized_state(
    token_ids: list[int],
    params: dict[str, Any] | None,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
) -> HiggsTtsState:
    """Build a Higgs state that uses caller-provided prompt ids verbatim."""
    params = params or {}
    return HiggsTtsState(
        prompt_token_ids=list(token_ids),
        reference_codes_delayed=None,
        reference_waveform=None,
        reference_code_cache_key=None,
        target_text=None,
        reference_text=None,
        uploaded_voice_name=None,
        uploaded_voice_created_at=None,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=int(params.get("max_new_tokens", 2048)),
        temperature=float(params.get("temperature", 1.0)),
        top_p=params.get("top_p"),
        top_k=params.get("top_k"),
        seed=params.get("seed"),
        return_logprob=bool(params.get("return_logprob", False)),
        return_omni_rollout=bool(params.get("return_omni_rollout", False)),
    )


__all__ = ["build_pretokenized_state", "is_pretokenized_prompt"]
