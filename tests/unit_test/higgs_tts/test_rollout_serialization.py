# SPDX-License-Identifier: Apache-2.0
"""apply_higgs_result -> HiggsTtsState.omni_rollout -> to_dict serialization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import apply_higgs_result
from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    delay_pattern_codec_content_mask,
)

N = 8
V = 1026


def _fake_data(*, return_omni_rollout, return_logprob, t_raw=6):
    delayed = apply_delay_pattern(torch.randint(0, 1024, (t_raw, N)))
    cb0_logprobs = [
        [float(-100.0 - i), int(code[0].item())]
        for i, code in enumerate(delayed.unbind(0))
    ]
    action_mask = delay_pattern_codec_content_mask(delayed)
    action_mask[t_raw, 0] = True
    return SimpleNamespace(
        output_codes=list(delayed.unbind(0)),
        output_action_masks=list(action_mask.unbind(0)),
        output_logprobs=list(torch.randn(*delayed.shape).unbind(0)),
        output_token_logprobs=cb0_logprobs,
        num_codebooks=N,
        codebook_size=V,
        return_omni_rollout=return_omni_rollout,
        return_logprob=return_logprob,
        input_ids=list(range(5)),
        weight_version="7",
    )


def test_omni_rollout_built_and_roundtrips():
    torch.manual_seed(0)
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)
    data = _fake_data(return_omni_rollout=True, return_logprob=True)
    apply_higgs_result(state, data)

    stream = state.omni_rollout["action_streams"][0]
    assert stream["name"] == "higgs_codes"
    assert stream["policy_logprobs"] is not None
    assert state.omni_rollout["version"] == 2
    assert state.omni_rollout["total_action_count"] == 6 * N + 1
    assert state.output_token_logprobs == data.output_token_logprobs
    assert state.weight_version == "7"
    # Survives the StagePayload dict round-trip the client reads from.
    assert HiggsTtsState.from_dict(state.to_dict()).omni_rollout == state.omni_rollout
    assert HiggsTtsState.from_dict(state.to_dict()).weight_version == "7"
    assert (
        HiggsTtsState.from_dict(state.to_dict()).output_token_logprobs
        == state.output_token_logprobs
    )


def test_flag_gating():
    torch.manual_seed(1)
    # no rollout flag -> nothing emitted.
    off = HiggsTtsState(num_codebooks=N, codebook_size=V)
    apply_higgs_result(off, _fake_data(return_omni_rollout=False, return_logprob=True))
    assert off.omni_rollout is None
    assert "omni_rollout" not in off.to_dict()
    assert off.output_token_logprobs is not None

    # A structured rollout without full policy logprobs is invalid.
    no_lp = HiggsTtsState(num_codebooks=N, codebook_size=V)
    with pytest.raises(ValueError, match="missing aligned"):
        apply_higgs_result(
            no_lp, _fake_data(return_omni_rollout=True, return_logprob=False)
        )


def test_cb0_output_logprobs_do_not_require_omni_rollout():
    torch.manual_seed(2)
    data = _fake_data(return_omni_rollout=False, return_logprob=True, t_raw=3)
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)

    apply_higgs_result(state, data)

    assert state.omni_rollout is None
    assert state.output_token_logprobs == data.output_token_logprobs


def test_cb0_output_logprob_shape_mismatch_fails_loudly():
    torch.manual_seed(3)
    data = _fake_data(return_omni_rollout=False, return_logprob=True, t_raw=3)
    data.output_token_logprobs = data.output_token_logprobs[:-1]
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)

    with pytest.raises(ValueError, match="output_token_logprobs length"):
        apply_higgs_result(state, data)


def test_cb0_output_logprob_token_mismatch_fails_loudly():
    torch.manual_seed(4)
    data = _fake_data(return_omni_rollout=False, return_logprob=True, t_raw=3)
    data.output_token_logprobs[0][1] += 1
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)

    with pytest.raises(ValueError, match="does not match codebook-0 token"):
        apply_higgs_result(state, data)


def test_rollout_logprob_shape_mismatch_fails_loudly():
    torch.manual_seed(5)
    data = _fake_data(return_omni_rollout=True, return_logprob=True, t_raw=3)
    data.output_logprobs = data.output_logprobs[:-1]
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)

    with pytest.raises(ValueError, match="rollout logprob shape"):
        apply_higgs_result(state, data)
