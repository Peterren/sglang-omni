# SPDX-License-Identifier: Apache-2.0
"""Tests for Higgs TTS piecewise prefill CUDA-graph enablement.

PCG is opt-in for Higgs: the engine builder re-enables SGLang's piecewise
CUDA graph (which ``ServerArgs.__post_init__`` disables for multimodal
models) and rebuilds the token buckets. ``--piecewise-cuda-graph`` toggles
it from the CLI; omitting the flag keeps the pipeline default (off).
"""

from __future__ import annotations

import inspect

import pytest
import typer

from sglang_omni.cli.serve import apply_piecewise_cuda_graph_cli_overrides
from sglang_omni.config import PipelineConfig, resolve_stage_factory_args
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.higgs_tts.engine_builder import HiggsTtsEngineBuilder
from sglang_omni.models.higgs_tts.stages import create_sglang_tts_engine_executor
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig


def _tts_engine_args(config: PipelineConfig) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == "tts_engine")
    return resolve_stage_factory_args(stage, config)


def _make_builder(*, enable_piecewise_cuda_graph: bool) -> HiggsTtsEngineBuilder:
    return HiggsTtsEngineBuilder(
        max_new_tokens=2048,
        max_running_requests=64,
        cuda_graph_max_bs=64,
        enable_async_decode=True,
        async_decode_min_batch_size=2,
        enable_piecewise_cuda_graph=enable_piecewise_cuda_graph,
    )


class _StubServerArgs:
    """The ServerArgs surface customize_server_args touches, post-__post_init__.

    Mirrors the state after SGLang's multimodal gate ran: PCG disabled and
    token buckets ungenerated.
    """

    def __init__(self) -> None:
        self.disable_overlap_schedule = False
        self.enforce_piecewise_cuda_graph = False
        self.disable_piecewise_cuda_graph = True
        self.piecewise_cuda_graph_max_tokens = None
        self.piecewise_cuda_graph_tokens = None

    def _generate_piecewise_cuda_graph_tokens(self) -> list[int]:
        assert self.piecewise_cuda_graph_max_tokens is not None
        return list(range(64, self.piecewise_cuda_graph_max_tokens + 1, 64))


def test_factory_defaults_to_pcg_off() -> None:
    signature = inspect.signature(create_sglang_tts_engine_executor)
    assert signature.parameters["enable_piecewise_cuda_graph"].default is False


def test_customize_server_args_enables_pcg() -> None:
    server_args = _StubServerArgs()
    _make_builder(enable_piecewise_cuda_graph=True).customize_server_args(server_args)

    assert server_args.enforce_piecewise_cuda_graph is True
    assert server_args.disable_piecewise_cuda_graph is False
    assert (
        server_args.piecewise_cuda_graph_max_tokens
        == HiggsTtsEngineBuilder.piecewise_cuda_graph_max_tokens
    )
    assert server_args.piecewise_cuda_graph_tokens
    assert max(server_args.piecewise_cuda_graph_tokens) == (
        HiggsTtsEngineBuilder.piecewise_cuda_graph_max_tokens
    )


def test_customize_server_args_leaves_pcg_disabled_by_default() -> None:
    server_args = _StubServerArgs()
    _make_builder(enable_piecewise_cuda_graph=False).customize_server_args(server_args)

    assert server_args.enforce_piecewise_cuda_graph is False
    assert server_args.disable_piecewise_cuda_graph is True
    assert server_args.piecewise_cuda_graph_tokens is None


def test_cli_override_toggles_factory_arg() -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")

    apply_piecewise_cuda_graph_cli_overrides(config, piecewise_cuda_graph=True)
    assert _tts_engine_args(config)["enable_piecewise_cuda_graph"] is True

    apply_piecewise_cuda_graph_cli_overrides(config, piecewise_cuda_graph=False)
    assert _tts_engine_args(config)["enable_piecewise_cuda_graph"] is False


def test_cli_override_none_is_noop() -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_piecewise_cuda_graph_cli_overrides(config, piecewise_cuda_graph=None)
    assert "enable_piecewise_cuda_graph" not in _tts_engine_args(config)


def test_cli_override_rejects_unsupported_pipeline() -> None:
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_piecewise_cuda_graph_cli_overrides(config, piecewise_cuda_graph=True)
