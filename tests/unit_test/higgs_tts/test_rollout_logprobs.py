# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for :func:`selected_token_logprobs` (RL rollout logprobs).

The contract these pin down (the train/rollout replay-equivalence gate):

- A sampled row's selected-action logprob is the FULL-vocab
  ``log_softmax(logits / temperature)`` at the sampled code -- top-k / top-p are
  sampling filters and must NOT truncate the returned distribution.
- A greedy row (``temperature <= 1e-5`` or ``top_k == 1``) uses
  ``log_softmax`` of the RAW logits (no temperature scaling).

All tests run on CPU; ``selected_token_logprobs`` uses only ``log_softmax`` +
``gather`` (no fused CUDA kernels).
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.sampler import (
    _GREEDY_TEMP_THRESHOLD,
    selected_token_logprobs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 8
V = 1026


def _manual_selected_logprob(
    logits_NV: torch.Tensor, codes_N: torch.Tensor, temp: float
) -> torch.Tensor:
    """Reference: full-vocab log_softmax(logits/temp) gathered at codes."""
    lp = torch.log_softmax(logits_NV.float() / temp, dim=-1)
    return lp.gather(-1, codes_N.long().unsqueeze(-1)).squeeze(-1)


def test_matches_manual_log_softmax_sampling():
    """Sampled rows: exact full-vocab log_softmax(logits/temp) at the code."""
    torch.manual_seed(0)
    B = 4
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temp_val = 0.7
    temperature = torch.full((B,), temp_val, device=DEVICE)

    got = selected_token_logprobs(logits, codes, temperature=temperature)

    expected = torch.stack(
        [_manual_selected_logprob(logits[b], codes[b], temp_val) for b in range(B)]
    )
    assert got.shape == (B, N)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


def test_is_a_valid_log_distribution():
    """exp(logprob) over the whole vocab sums to 1 per (row, codebook)."""
    torch.manual_seed(1)
    B = 3
    logits = torch.randn(B, N, V, device=DEVICE)
    temperature = torch.full((B,), 1.3, device=DEVICE)

    # Selected logprob for EVERY vocab id must reconstruct a normalized dist.
    all_codes = torch.arange(V, device=DEVICE)
    # Gather logprob of each vocab id at a single (row, codebook) cell.
    logits_cell = logits[0, 0].view(1, 1, V).expand(V, 1, V)
    temp_cell = torch.full((V,), 1.3, device=DEVICE)
    lp = selected_token_logprobs(
        logits_cell, all_codes.view(V, 1), temperature=temp_cell
    )[:, 0]
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.0, device=DEVICE), atol=1e-4)


def test_temperature_scaling_changes_value_per_row():
    """Per-row temperature is applied independently."""
    torch.manual_seed(2)
    B = 2
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.tensor([0.5, 2.0], device=DEVICE)

    got = selected_token_logprobs(logits, codes, temperature=temperature)

    exp0 = _manual_selected_logprob(logits[0], codes[0], 0.5)
    exp1 = _manual_selected_logprob(logits[1], codes[1], 2.0)
    assert torch.allclose(got[0], exp0, atol=1e-5)
    assert torch.allclose(got[1], exp1, atol=1e-5)
    # Different temperatures give different logprobs on the same logits/codes.
    assert not torch.allclose(got[0], got[1])


def test_topk_topp_do_not_truncate_logprob():
    """KEY contract: top-k>1 must NOT change the returned full-vocab logprob.

    Sampling filters change WHICH token is drawn, never the policy probability
    assigned to it for the loss. A non-unit top_k must leave the logprob equal to
    the unfiltered full-vocab log_softmax.
    """
    torch.manual_seed(3)
    B = 2
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.full((B,), 1.0, device=DEVICE)

    base = selected_token_logprobs(logits, codes, temperature=temperature)
    with_topk = selected_token_logprobs(
        logits,
        codes,
        temperature=temperature,
        top_k_buf=torch.full((B,), 5, dtype=torch.long, device=DEVICE),
    )
    assert torch.allclose(base, with_topk, atol=0.0)


def test_greedy_temperature_uses_raw_logits():
    """A greedy row (temp ~ 0) returns log_softmax of the RAW logits."""
    torch.manual_seed(4)
    B = 1
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.full((B,), 0.0, device=DEVICE)

    got = selected_token_logprobs(logits, codes, temperature=temperature)

    expected = torch.log_softmax(logits[0].float(), dim=-1)
    expected = expected.gather(-1, codes[0].long().unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got[0], expected, atol=1e-5)


def test_top_k_one_uses_greedy_convention():
    """``top_k == 1`` forces the greedy (raw-logit) convention even at temp=1."""
    torch.manual_seed(5)
    B = 1
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.full((B,), 1.0, device=DEVICE)
    top_k_buf = torch.ones(B, dtype=torch.long, device=DEVICE)

    got = selected_token_logprobs(
        logits, codes, temperature=temperature, top_k_buf=top_k_buf
    )

    expected = torch.log_softmax(logits[0].float(), dim=-1)
    expected = expected.gather(-1, codes[0].long().unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got[0], expected, atol=1e-5)
    # Sanity: this differs from the temp-scaled value unless temp==1 collapses it.
    sampled = _manual_selected_logprob(logits[0], codes[0], 1.0)
    assert torch.allclose(got[0], sampled, atol=1e-5)  # temp=1 => raw == scaled


def test_mixed_greedy_and_sampled_rows():
    """A batch mixing greedy and sampled rows resolves each by its own rule."""
    torch.manual_seed(6)
    B = 4
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.tensor([0.0, 1.5, 0.0, 0.8], device=DEVICE)

    got = selected_token_logprobs(logits, codes, temperature=temperature)

    for b, temp_val in enumerate([0.0, 1.5, 0.0, 0.8]):
        if temp_val <= _GREEDY_TEMP_THRESHOLD:
            lp = torch.log_softmax(logits[b].float(), dim=-1)
            exp_b = lp.gather(-1, codes[b].long().unsqueeze(-1)).squeeze(-1)
        else:
            exp_b = _manual_selected_logprob(logits[b], codes[b], temp_val)
        assert torch.allclose(got[b], exp_b, atol=1e-5), f"row {b}"


def test_dtype_is_float32_even_for_bf16_logits():
    """Logits may arrive bf16; logprobs are computed/returned in float32."""
    B = 2
    logits = torch.randn(B, N, V, device=DEVICE, dtype=torch.bfloat16)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.full((B,), 1.0, device=DEVICE)

    got = selected_token_logprobs(logits, codes, temperature=temperature)
    assert got.dtype == torch.float32


def test_out_of_range_code_raises():
    """A STOP/-1 (or otherwise invalid) code must fail loud, not gather garbage."""
    # CUDA turns an out-of-bounds gather into an async device-side assert that
    # poisons the context for later tests, so assert the loud failure on CPU only.
    if DEVICE != "cpu":
        pytest.skip("out-of-bounds gather is an unrecoverable device assert on CUDA")
    B = 1
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.full((B, N), -1, dtype=torch.long, device=DEVICE)
    temperature = torch.full((B,), 1.0, device=DEVICE)

    with pytest.raises((RuntimeError, IndexError)):
        selected_token_logprobs(logits, codes, temperature=temperature)
