# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`build_omni_rollout_trace` (meta_info.omni_rollout builder)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.rollout_trace import (
    OMNI_ROLLOUT_VERSION,
    build_omni_rollout_trace,
)
from sglang_omni.models.higgs_tts.utils import apply_delay_pattern

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 8
V = 1026


def _delayed(t_raw: int, n: int = N) -> torch.Tensor:
    raw = torch.randint(0, 1024, (t_raw, n), device=DEVICE)
    return apply_delay_pattern(raw)


def test_schema_shape_and_labels():
    torch.manual_seed(0)
    delayed = _delayed(10)
    lp = torch.randn(*delayed.shape, device=DEVICE)

    trace = build_omni_rollout_trace(
        delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=lp
    )

    assert trace["version"] == OMNI_ROLLOUT_VERSION
    assert trace["model_family"] == "higgs_tts"
    assert trace["stages"] == ["tts_engine"]
    assert trace["non_action_outputs"] == []
    assert len(trace["action_streams"]) == 1

    s = trace["action_streams"][0]
    assert s["name"] == "higgs_codes"
    assert s["layout"] == "codebook_2d"
    assert s["action_type"] == "discrete"
    assert s["shape"] == list(delayed.shape)
    assert s["vocab_size"] == V
    assert s["channel_ids"] == list(range(N))
    assert s["channel_roles"] == [f"codebook_{c}" for c in range(N)]


def test_actions_match_input_codes():
    torch.manual_seed(1)
    delayed = _delayed(6)
    trace = build_omni_rollout_trace(delayed, num_codebooks=N, codebook_vocab_size=V)
    s = trace["action_streams"][0]
    assert s["actions"] == delayed.to(torch.long).tolist()
    assert s["logprobs"] is None  # not requested


def test_total_action_count_equals_mask_sum_equals_T_times_N():
    torch.manual_seed(2)
    t_raw = 12
    delayed = _delayed(t_raw)
    trace = build_omni_rollout_trace(delayed, num_codebooks=N, codebook_vocab_size=V)
    s = trace["action_streams"][0]

    mask_sum = sum(sum(row) for row in s["action_mask"])
    assert trace["total_action_count"] == mask_sum
    assert trace["total_action_count"] == t_raw * N


def test_action_mask_is_parallelogram_inverse_of_specials():
    """action_mask==1 exactly where the delayed code is real audio (not BOC/EOC)."""
    torch.manual_seed(3)
    delayed = _delayed(9)
    trace = build_omni_rollout_trace(delayed, num_codebooks=N, codebook_vocab_size=V)
    mask = torch.tensor(trace["action_streams"][0]["action_mask"], device=DEVICE).bool()
    is_real = (delayed != 1024) & (delayed != 1025)  # BOC=1024, EOC=1025
    assert torch.equal(mask, is_real)


def test_logprobs_serialized_aligned():
    torch.manual_seed(4)
    delayed = _delayed(5)
    lp = torch.randn(*delayed.shape, device=DEVICE)
    trace = build_omni_rollout_trace(
        delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=lp
    )
    got = torch.tensor(trace["action_streams"][0]["logprobs"], device=DEVICE)
    assert torch.allclose(got, lp, atol=1e-5)


def test_empty_generation():
    """Zero rows generated -> empty stream, no actions."""
    delayed = torch.empty((0, N), dtype=torch.long, device=DEVICE)
    trace = build_omni_rollout_trace(delayed, num_codebooks=N, codebook_vocab_size=V)
    s = trace["action_streams"][0]
    assert s["shape"] == [0, N]
    assert s["actions"] == []
    assert trace["total_action_count"] == 0


def test_codebook_count_mismatch_raises():
    delayed = _delayed(4, n=N)
    with pytest.raises(ValueError, match="codebooks"):
        build_omni_rollout_trace(delayed, num_codebooks=N + 1, codebook_vocab_size=V)


def test_logprob_shape_mismatch_raises():
    delayed = _delayed(4)
    bad_lp = torch.randn(delayed.shape[0] + 1, N, device=DEVICE)
    with pytest.raises(ValueError, match="shape"):
        build_omni_rollout_trace(
            delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=bad_lp
        )


def test_non_finite_logprob_at_action_raises():
    """A NaN logprob on a real-action cell must fail loud."""
    torch.manual_seed(5)
    delayed = _delayed(6)
    lp = torch.randn(*delayed.shape, device=DEVICE)
    # Cell (3, 0) is a real action (cb0 within [0, T)); poison it.
    lp[3, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_omni_rollout_trace(
            delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=lp
        )


def test_non_finite_logprob_on_masked_cell_is_ok():
    """A non-finite value on a NON-action (scaffolding) cell is harmless."""
    torch.manual_seed(6)
    delayed = _delayed(6)
    lp = torch.randn(*delayed.shape, device=DEVICE)
    lp[0, N - 1] = float("-inf")  # leading BOC triangle cell -> masked out
    trace = build_omni_rollout_trace(
        delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=lp
    )
    assert trace["total_action_count"] > 0
