# SPDX-License-Identifier: Apache-2.0
"""Build the structured Higgs policy action trace."""

from __future__ import annotations

from typing import Any

import torch

ACTION_TRACE_VERSION = 1


def build_action_trace(
    delayed_codes: torch.Tensor,
    *,
    num_codebooks: int,
    codebook_vocab_size: int,
    policy_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
    stage: str = "tts_engine",
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
    stream: dict[str, Any] = {
        "stage": stage,
        "modality": "audio",
        "vocab_size": int(codebook_vocab_size),
        "actions": delayed_codes.to(torch.long).tolist(),
        "logprobs": policy_logprobs.tolist(),
        "action_mask": action_mask.tolist(),
    }

    return {
        "version": ACTION_TRACE_VERSION,
        "streams": [stream],
    }


__all__ = ["ACTION_TRACE_VERSION", "build_action_trace"]
