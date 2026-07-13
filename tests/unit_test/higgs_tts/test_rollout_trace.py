# SPDX-License-Identifier: Apache-2.0
"""Higgs policy action-trace validation and serialization."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from sglang_omni.models.higgs_tts.rollout_trace import (
    ACTION_TRACE_VERSION,
    build_action_trace,
)
from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    delay_pattern_codec_content_mask,
)
from sglang_omni.serve.protocol import ActionTrace

N = 8
V = 1026


def _inputs(t_raw: int = 4):
    codes = apply_delay_pattern(torch.randint(0, 1024, (t_raw, N)))
    mask = delay_pattern_codec_content_mask(codes)
    mask[t_raw, 0] = True  # sampled terminating EOC is an action, not content
    logprobs = torch.randn(codes.shape)
    return codes, mask, logprobs


def test_schema_uses_explicit_sample_time_mask():
    codes, mask, logprobs = _inputs()
    trace = build_action_trace(
        codes,
        num_codebooks=N,
        codebook_vocab_size=V,
        policy_logprobs=logprobs,
        action_mask=mask,
    )

    parsed = ActionTrace.model_validate(trace)
    assert parsed.version == ACTION_TRACE_VERSION == 1
    stream = trace["streams"][0]
    assert stream["actions"] == codes.tolist()
    assert stream["action_mask"] == mask.tolist()
    serialized_lp = torch.tensor(stream["logprobs"])
    assert torch.all(serialized_lp[~mask] == 0)
    assert torch.allclose(serialized_lp[mask], logprobs[mask])


def test_input_guards_and_nonfinite_masking():
    codes, mask, logprobs = _inputs()
    kwargs = dict(
        num_codebooks=N,
        codebook_vocab_size=V,
        policy_logprobs=logprobs,
        action_mask=mask,
    )
    with pytest.raises(ValueError, match="codebooks"):
        build_action_trace(codes, **{**kwargs, "num_codebooks": N + 1})
    with pytest.raises(ValueError, match="action_mask shape"):
        build_action_trace(codes, **{**kwargs, "action_mask": mask[:-1]})
    with pytest.raises(ValueError, match="policy_logprobs shape"):
        build_action_trace(codes, **{**kwargs, "policy_logprobs": logprobs[:-1]})

    row, channel = mask.nonzero()[0]
    logprobs[row, channel] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_action_trace(codes, **kwargs)
    codes[0, 0] = V
    with pytest.raises(ValueError, match="vocabulary"):
        build_action_trace(codes, **kwargs)


def test_wire_schema_rejects_integer_masks_and_unknown_fields():
    codes, mask, logprobs = _inputs()
    trace = build_action_trace(
        codes,
        num_codebooks=N,
        codebook_vocab_size=V,
        policy_logprobs=logprobs,
        action_mask=mask,
    )
    trace["streams"][0]["action_mask"][0][0] = 1
    with pytest.raises(ValidationError):
        ActionTrace.model_validate(trace)

    trace = build_action_trace(
        codes,
        num_codebooks=N,
        codebook_vocab_size=V,
        policy_logprobs=logprobs,
        action_mask=mask,
    )
    trace["streams"][0]["unexpected"] = "must fail closed"
    with pytest.raises(ValidationError):
        ActionTrace.model_validate(trace)
