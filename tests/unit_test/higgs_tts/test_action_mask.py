# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`delay_pattern_action_mask` (RL trainable-action mask).

The mask must select exactly the ``T x N`` real-audio parallelogram of a delayed
code matrix -- the inverse geometry of :func:`apply_delay_pattern` -- excluding
leading BOC scaffolding and trailing EOC wind-down padding.

All tests run on CPU.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.utils import (
    BOC_ID,
    EOC_ID,
    apply_delay_pattern,
    delay_pattern_action_mask,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("t_raw,n", [(1, 8), (3, 4), (10, 8), (5, 2), (1, 1), (20, 3)])
def test_mask_is_inverse_of_apply_delay_pattern(t_raw: int, n: int):
    """The mask equals exactly the non-BOC/EOC cells of a delayed matrix.

    ``apply_delay_pattern`` fills the real-audio parallelogram with codes in
    ``[0, 1023]`` and everything else with BOC/EOC, so "is an action" is
    equivalent to "is not a special sentinel".
    """
    torch.manual_seed(t_raw * 100 + n)
    raw_TN = torch.randint(0, 1024, (t_raw, n), device=DEVICE)  # real codes only
    delayed = apply_delay_pattern(raw_TN)
    assert delayed.shape == (t_raw + n - 1, n)

    mask = delay_pattern_action_mask(delayed)

    is_real = (delayed != BOC_ID) & (delayed != EOC_ID)
    assert torch.equal(mask, is_real)
    assert int(mask.sum().item()) == t_raw * n


@pytest.mark.parametrize("t_raw,n", [(1, 8), (3, 4), (10, 8), (5, 2), (20, 3)])
def test_mask_geometry_parallelogram(t_raw: int, n: int):
    """Position (r, c) is an action iff ``c <= r < c + T``."""
    torch.manual_seed(7)
    raw_TN = torch.randint(0, 1024, (t_raw, n), device=DEVICE)
    delayed = apply_delay_pattern(raw_TN)
    L = t_raw + n - 1

    mask = delay_pattern_action_mask(delayed)

    r = torch.arange(L, device=DEVICE).unsqueeze(1)
    c = torch.arange(n, device=DEVICE).unsqueeze(0)
    expected = (c <= r) & (r < c + t_raw)
    assert torch.equal(mask, expected)


def test_mask_explicit_small_case():
    """Hand-verified T=3, N=4 mask (L = 6)."""
    raw_TN = torch.arange(12, device=DEVICE).reshape(3, 4)  # codes 0..11, all real
    delayed = apply_delay_pattern(raw_TN)  # [6, 4]

    mask = delay_pattern_action_mask(delayed)

    # Rows r=0..5, codebooks c=0..3; action iff c <= r < c+3.
    expected = torch.tensor(
        [
            [1, 0, 0, 0],  # r=0: only c=0
            [1, 1, 0, 0],  # r=1: c=0,1
            [1, 1, 1, 0],  # r=2: c=0,1,2
            [0, 1, 1, 1],  # r=3: c=1,2,3 (c=0 done: 0<=3<3 is False)
            [0, 0, 1, 1],  # r=4: c=2,3
            [0, 0, 0, 1],  # r=5: c=3
        ],
        dtype=torch.bool,
        device=DEVICE,
    )
    assert torch.equal(mask, expected)


def test_T_recovered_from_cb0_eoc_only():
    """``T`` comes from codebook 0's EOC, not an EOC that appears in another cb."""
    n = 4
    t_raw = 5
    raw_TN = torch.randint(0, 1024, (t_raw, n), device=DEVICE)
    delayed = apply_delay_pattern(raw_TN)
    # Inject a spurious EOC in codebook 2 at an early row; cb0's EOC (row T=5)
    # must still define the parallelogram.
    delayed[1, 2] = EOC_ID

    mask = delay_pattern_action_mask(delayed)

    r = torch.arange(delayed.shape[0], device=DEVICE).unsqueeze(1)
    c = torch.arange(n, device=DEVICE).unsqueeze(0)
    expected = (c <= r) & (r < c + t_raw)
    assert torch.equal(mask, expected)


def test_no_cb0_eoc_means_T_equals_L():
    """Length-truncated generation (cb0 never emits EOC): every post-delay cell
    after the leading BOC triangle is an action (``T = L``)."""
    n = 3
    L = 5
    # All real codes, leading BOC delay triangle, NO EOC anywhere.
    delayed = torch.randint(0, 1024, (L, n), device=DEVICE)
    for c in range(n):
        delayed[:c, c] = BOC_ID  # leading delay triangle

    mask = delay_pattern_action_mask(delayed)

    r = torch.arange(L, device=DEVICE).unsqueeze(1)
    c = torch.arange(n, device=DEVICE).unsqueeze(0)
    expected = c <= r  # r < c + L always holds since r < L <= c + L
    assert torch.equal(mask, expected)


def test_cb0_eoc_row_itself_is_not_an_action():
    """The terminal cb0 EOC (stop marker) is excluded from the action set."""
    n = 4
    t_raw = 6
    raw_TN = torch.randint(0, 1024, (t_raw, n), device=DEVICE)
    delayed = apply_delay_pattern(raw_TN)

    mask = delay_pattern_action_mask(delayed)

    # cb0 EOC sits at row t_raw; it must be masked out.
    assert bool(delayed[t_raw, 0].item() == EOC_ID)
    assert not bool(mask[t_raw, 0].item())
    # And the last real cb0 action is row t_raw - 1.
    assert bool(mask[t_raw - 1, 0].item())


def test_mask_dtype_and_device():
    raw_TN = torch.randint(0, 1024, (4, 3), device=DEVICE)
    delayed = apply_delay_pattern(raw_TN)
    mask = delay_pattern_action_mask(delayed)
    assert mask.dtype == torch.bool
    assert mask.device.type == torch.device(DEVICE).type
    assert mask.shape == delayed.shape


def test_non_2d_raises():
    with pytest.raises(ValueError, match="2-D"):
        delay_pattern_action_mask(torch.zeros(4, device=DEVICE))
    with pytest.raises(ValueError, match="2-D"):
        delay_pattern_action_mask(torch.zeros(2, 3, 4, device=DEVICE))
