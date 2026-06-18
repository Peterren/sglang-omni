# SPDX-License-Identifier: Apache-2.0
"""apply_higgs_result -> HiggsTtsState.omni_rollout -> to_dict serialization."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import apply_higgs_result
from sglang_omni.models.higgs_tts.utils import apply_delay_pattern

N = 8
V = 1026


def _fake_data(*, return_omni_rollout, return_logprob, t_raw=6, with_codes=True):
    if with_codes:
        delayed = apply_delay_pattern(torch.randint(0, 1024, (t_raw, N)))
        output_codes = list(delayed.unbind(0))
        output_logprobs = list(torch.randn(*delayed.shape).unbind(0))
    else:
        output_codes, output_logprobs = [], []
    return SimpleNamespace(
        output_codes=output_codes,
        output_logprobs=output_logprobs,
        num_codebooks=N,
        codebook_size=V,
        return_omni_rollout=return_omni_rollout,
        return_logprob=return_logprob,
        input_ids=list(range(5)),
    )


def test_omni_rollout_built_and_roundtrips():
    torch.manual_seed(0)
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)
    data = _fake_data(return_omni_rollout=True, return_logprob=True)

    apply_higgs_result(state, data)

    assert state.omni_rollout is not None
    stream = state.omni_rollout["action_streams"][0]
    assert stream["name"] == "higgs_codes"
    assert stream["logprobs"] is not None
    assert state.omni_rollout["total_action_count"] == 6 * N

    # Survives the StagePayload dict round-trip the client reads from.
    restored = HiggsTtsState.from_dict(state.to_dict())
    assert restored.omni_rollout == state.omni_rollout


def test_no_rollout_flag_leaves_omni_rollout_none():
    torch.manual_seed(1)
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)
    data = _fake_data(return_omni_rollout=False, return_logprob=True)
    apply_higgs_result(state, data)
    assert state.omni_rollout is None
    assert "omni_rollout" not in state.to_dict()


def test_rollout_without_logprob_flag_omits_logprobs():
    torch.manual_seed(2)
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)
    data = _fake_data(return_omni_rollout=True, return_logprob=False)
    apply_higgs_result(state, data)
    assert state.omni_rollout["action_streams"][0]["logprobs"] is None


def test_empty_generation_still_emits_trace():
    """rollout requested but nothing generated -> empty but valid trace."""
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)
    data = _fake_data(return_omni_rollout=True, return_logprob=True, with_codes=False)
    apply_higgs_result(state, data)
    assert state.omni_rollout is not None
    assert state.omni_rollout["total_action_count"] == 0
    assert state.omni_rollout["action_streams"][0]["shape"] == [0, N]
