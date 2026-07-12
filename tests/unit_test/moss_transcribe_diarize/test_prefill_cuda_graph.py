# SPDX-License-Identifier: Apache-2.0
"""Tests for MOSS-TD piecewise prefill CUDA-graph integration."""

from __future__ import annotations

import inspect

import pytest
import torch

pytest.importorskip("sglang")

from sglang_omni.models.moss_transcribe_diarize.sglang_model import (  # noqa: E402
    MossTranscribeDiarizeForConditionalGeneration as MossModel,
)
from sglang_omni.models.moss_transcribe_diarize.stages import (  # noqa: E402
    create_sglang_moss_transcribe_diarize_executor,
)


def test_piecewise_cuda_graph_factory_defaults() -> None:
    signature = inspect.signature(create_sglang_moss_transcribe_diarize_executor)

    assert signature.parameters["enable_prefill_cuda_graph"].default is True
    assert signature.parameters["prefill_graph_token_buckets"].default is None


def test_piecewise_cuda_graph_model_alias_is_not_registered_twice() -> None:
    model = MossModel.__new__(MossModel)
    torch.nn.Module.__init__(model)
    language_model = torch.nn.Linear(4, 4, bias=False)
    model.language_model = language_model

    assert model.model is language_model
    assert "model" not in model._modules
    assert set(model.state_dict()) == {"language_model.weight"}

    model.model = torch.nn.Linear(4, 4, bias=False)

    assert model.model is language_model
    assert "model" not in model._modules
    assert set(model.state_dict()) == {"language_model.weight"}
