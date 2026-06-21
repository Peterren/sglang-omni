# SPDX-License-Identifier: Apache-2.0
"""Generation-stage batch policy validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.generation_batch_policy import (
    build_power_of_two_cuda_graph_bs,
    sync_cuda_graph_bs_with_max_bs,
    validate_generation_batch_policy,
)


def _server_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "max_running_requests": 16,
        "disable_cuda_graph": False,
        "cuda_graph_max_bs": 16,
        "cuda_graph_bs": [1, 2, 4, 8, 16],
        "torch_compile_max_bs": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_power_of_two_cuda_graph_bs_includes_requested_max() -> None:
    assert build_power_of_two_cuda_graph_bs(1) == [1]
    assert build_power_of_two_cuda_graph_bs(16) == [1, 2, 4, 8, 16]
    assert build_power_of_two_cuda_graph_bs(24) == [1, 2, 4, 8, 16, 24]


def test_validate_generation_batch_policy_reports_explicit_full_policy() -> None:
    report = validate_generation_batch_policy(
        model_name="test-model",
        server_args=_server_args(),
        model_buffer_bs=16,
    )

    assert report.max_running_requests == 16
    assert report.cuda_graph_enabled is True
    assert report.cuda_graph_max_bs == 16
    assert report.cuda_graph_bs == (1, 2, 4, 8, 16)
    assert report.torch_compile_max_bs == 16
    assert report.model_buffer_bs == 16


def test_validate_generation_batch_policy_rejects_implicit_cuda_graph_bs() -> None:
    with pytest.raises(ValueError, match="cuda_graph_bs must be explicit"):
        validate_generation_batch_policy(
            model_name="test-model",
            server_args=_server_args(cuda_graph_bs=None),
        )


def test_validate_generation_batch_policy_rejects_mismatched_cuda_graph_max() -> None:
    with pytest.raises(ValueError, match=r"max\(cuda_graph_bs\) must match"):
        validate_generation_batch_policy(
            model_name="test-model",
            server_args=_server_args(cuda_graph_max_bs=32),
        )


def test_validate_generation_batch_policy_requires_compile_coverage_or_exception() -> None:
    partial_compile = _server_args(
        max_running_requests=64,
        cuda_graph_max_bs=64,
        cuda_graph_bs=[1, 2, 4, 8, 16, 32, 64],
        torch_compile_max_bs=16,
    )
    with pytest.raises(ValueError, match="torch_compile_max_bs must cover"):
        validate_generation_batch_policy(
            model_name="test-model",
            server_args=partial_compile,
        )

    report = validate_generation_batch_policy(
        model_name="test-model",
        server_args=partial_compile,
        allow_partial_torch_compile_coverage=True,
    )
    assert report.torch_compile_max_bs == 16


def test_validate_generation_batch_policy_rejects_under_sized_model_buffer() -> None:
    with pytest.raises(ValueError, match="model_buffer_bs must cover"):
        validate_generation_batch_policy(
            model_name="test-model",
            server_args=_server_args(max_running_requests=4),
            model_buffer_bs=2,
        )


def test_sync_cuda_graph_bs_with_max_bs_preserves_explicit_list() -> None:
    overrides: dict[str, object] = {
        "cuda_graph_max_bs": 16,
        "cuda_graph_bs": [1, 2, 4, 8, 16],
    }
    server_args_overrides = {"cuda_graph_max_bs": 32, "cuda_graph_bs": [1, 4, 32]}
    overrides.update(server_args_overrides)
    sync_cuda_graph_bs_with_max_bs(overrides, server_args_overrides)
    assert overrides["cuda_graph_bs"] == [1, 4, 32]


def test_sync_cuda_graph_bs_with_max_bs_fills_missing_list() -> None:
    overrides: dict[str, object] = {"cuda_graph_max_bs": 32}
    sync_cuda_graph_bs_with_max_bs(overrides, {"cuda_graph_max_bs": 32})
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8, 16, 32]
