# SPDX-License-Identifier: Apache-2.0
"""Build the version-2 structured Higgs rollout trace."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.higgs_tts.utils import delay_pattern_codec_content_mask

OMNI_ROLLOUT_VERSION = 2


def build_omni_rollout_trace(
    delayed_codes: torch.Tensor,
    *,
    num_codebooks: int,
    codebook_vocab_size: int,
    policy_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
    model_family: str = "higgs_tts",
    stage: str = "tts_engine",
    stream_name: str = "higgs_codes",
) -> dict[str, Any]:
    """Validate and serialize aligned sampled actions, masks, and logprobs."""
    if delayed_codes.ndim != 2:
        raise ValueError(
            f"delayed_codes must be 2-D [L, N], got shape {tuple(delayed_codes.shape)}"
        )
    L, N = delayed_codes.shape
    if N != num_codebooks:
        raise ValueError(
            f"delayed_codes has {N} codebooks but num_codebooks={num_codebooks}"
        )

    if tuple(action_mask.shape) != (L, N):
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} != codes shape {(L, N)}"
        )
    if tuple(policy_logprobs.shape) != (L, N):
        raise ValueError(
            f"policy_logprobs shape {tuple(policy_logprobs.shape)} != "
            f"codes shape {(L, N)}"
        )
    action_mask = action_mask.to(torch.bool)
    if delayed_codes.numel() and not bool(
        ((delayed_codes >= 0) & (delayed_codes < codebook_vocab_size)).all()
    ):
        raise ValueError("Higgs rollout action is outside the codebook vocabulary")
    action_logprobs = policy_logprobs[action_mask]
    if action_logprobs.numel() and not bool(torch.isfinite(action_logprobs).all()):
        raise ValueError("non-finite policy logprob at a sampled action position")

    policy_logprobs = torch.where(
        action_mask,
        policy_logprobs.to(torch.float32),
        torch.zeros_like(policy_logprobs, dtype=torch.float32),
    )
    codec_content_mask = delay_pattern_codec_content_mask(delayed_codes)

    stream: dict[str, Any] = {
        "name": stream_name,
        "stage": stage,
        "modality": "audio",
        "action_type": "multi_discrete",
        "layout": "time_codebook",
        "shape": [int(L), int(N)],
        "vocab_size": int(codebook_vocab_size),
        "actions": delayed_codes.to(torch.long).tolist(),
        "policy_logprobs": policy_logprobs.tolist(),
        "action_mask": action_mask.tolist(),
        "codec_content_mask": codec_content_mask.tolist(),
        "channel_ids": list(range(N)),
    }

    return {
        "version": OMNI_ROLLOUT_VERSION,
        "model_family": model_family,
        "total_action_count": int(action_mask.sum().item()),
        "action_streams": [stream],
    }


__all__ = ["OMNI_ROLLOUT_VERSION", "build_omni_rollout_trace"]
