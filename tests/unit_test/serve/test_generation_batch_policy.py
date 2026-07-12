# SPDX-License-Identifier: Apache-2.0
"""Generation-stage batch policy validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.generation_batch_policy import (
    build_default_cuda_graph_bs,
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)


def _server_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "max_running_requests": 16,
        "disable_cuda_graph": False,
        "cuda_graph_max_bs": 16,
        "cuda_graph_bs": [1, 2, 4, 8, 12, 16],
        "enable_torch_compile": True,
        "torch_compile_max_bs": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_cuda_graph_bs_matches_sglang_normal_buckets() -> None:
    assert build_default_cuda_graph_bs(1) == [1]
    assert build_default_cuda_graph_bs(16) == [1, 2, 4, 8, 12, 16]
    assert build_default_cuda_graph_bs(24) == [1, 2, 4, 8, 12, 16, 24]
    assert build_default_cuda_graph_bs(64) == [
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    ]


def test_build_generation_batch_overrides_tie_batch_knobs() -> None:
    assert build_generation_batch_overrides(max_running_requests=16) == {
        "disable_piecewise_cuda_graph": False,
        "max_running_requests": 16,
        "cuda_graph_max_bs": 16,
        "torch_compile_max_bs": 16,
        "cuda_graph_bs": [1, 2, 4, 8, 12, 16],
    }


def test_build_generation_batch_overrides_allow_explicit_caps() -> None:
    assert build_generation_batch_overrides(
        max_running_requests=16,
        cuda_graph_max_bs=32,
        torch_compile_max_bs=8,
    ) == {
        "disable_piecewise_cuda_graph": False,
        "max_running_requests": 16,
        "cuda_graph_max_bs": 32,
        "torch_compile_max_bs": 8,
        "cuda_graph_bs": [1, 2, 4, 8, 12, 16, 24, 32],
    }


def test_build_generation_batch_overrides_reject_non_positive_values() -> None:
    with pytest.raises(ValueError, match="max_running_requests"):
        build_generation_batch_overrides(max_running_requests=0)
    with pytest.raises(ValueError, match="cuda_graph_max_bs"):
        build_generation_batch_overrides(max_running_requests=1, cuda_graph_max_bs=0)
    with pytest.raises(ValueError, match="torch_compile_max_bs"):
        build_generation_batch_overrides(max_running_requests=1, torch_compile_max_bs=0)


def test_validate_generation_batch_policy_accepts_explicit_full_policy() -> None:
    validate_generation_batch_policy(
        model_name="test-model",
        server_args=_server_args(),
        model_buffer_bs=16,
    )


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


def test_validate_generation_batch_policy_requires_enabled_compile_coverage() -> None:
    undercovered_compile = _server_args(
        max_running_requests=64,
        cuda_graph_max_bs=64,
        cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
        torch_compile_max_bs=16,
    )
    with pytest.raises(ValueError, match="torch_compile_max_bs must cover"):
        validate_generation_batch_policy(
            model_name="test-model",
            server_args=undercovered_compile,
        )


def test_validate_generation_batch_policy_ignores_disabled_compile_cap() -> None:
    validate_generation_batch_policy(
        model_name="test-model",
        server_args=_server_args(
            max_running_requests=64,
            cuda_graph_max_bs=64,
            cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
            enable_torch_compile=False,
            torch_compile_max_bs=16,
        ),
    )


def test_validate_generation_batch_policy_rejects_under_sized_model_buffer() -> None:
    with pytest.raises(ValueError, match="model_buffer_bs must cover"):
        validate_generation_batch_policy(
            model_name="test-model",
            server_args=_server_args(max_running_requests=4),
            model_buffer_bs=2,
        )


def test_build_generation_batch_overrides_preserves_explicit_list() -> None:
    server_args_overrides = {"cuda_graph_max_bs": 32, "cuda_graph_bs": [1, 4, 32]}
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        server_args_overrides=server_args_overrides,
    )
    assert overrides["cuda_graph_bs"] == [1, 4, 32]


def test_build_generation_batch_overrides_fills_default_list() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        cuda_graph_max_bs=32,
    )
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16, 24, 32]


def test_build_generation_batch_overrides_recomputes_list_when_max_changes() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        server_args_overrides={"cuda_graph_max_bs": 32},
    )
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16, 24, 32]


def test_build_generation_batch_overrides_rebinds_default_caps_when_max_changes() -> (
    None
):
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        server_args_overrides={"max_running_requests": 32},
    )
    assert overrides["max_running_requests"] == 32
    assert overrides["cuda_graph_max_bs"] == 32
    assert overrides["torch_compile_max_bs"] == 32
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16, 24, 32]


def test_build_generation_batch_overrides_uses_sglang_prefill_graph_sizes() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        enable_prefill_cuda_graph=True,
        prefill_graph_token_buckets=None,
        server_args_overrides={"disable_radix_cache": True},
    )

    assert overrides["disable_radix_cache"] is True
    assert overrides["disable_piecewise_cuda_graph"] is False
    assert "piecewise_cuda_graph_tokens" not in overrides


def test_build_generation_batch_overrides_disables_prefill_graph() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        enable_prefill_cuda_graph=False,
        prefill_graph_token_buckets=None,
    )

    assert overrides["disable_piecewise_cuda_graph"] is True
    assert "piecewise_cuda_graph_tokens" not in overrides


def test_build_generation_batch_overrides_normalizes_prefill_graph_sizes() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        enable_prefill_cuda_graph=True,
        prefill_graph_token_buckets=[8192, 240, 4608, 240],
        server_args_overrides={"chunked_prefill_size": 8192},
    )

    assert overrides["chunked_prefill_size"] == 8192
    assert overrides["disable_piecewise_cuda_graph"] is False
    assert overrides["piecewise_cuda_graph_tokens"] == [240, 4608, 8192]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("disable_piecewise_cuda_graph", True),
        ("enforce_piecewise_cuda_graph", True),
        ("piecewise_cuda_graph_tokens", [240, 640]),
    ],
)
def test_build_generation_batch_overrides_preserves_low_level_prefill_graph(
    key: str, value: object
) -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=16,
        enable_prefill_cuda_graph=True,
        prefill_graph_token_buckets=None,
        server_args_overrides={key: value},
    )

    assert overrides[key] == value


def test_build_generation_batch_overrides_rejects_buckets_when_disabled() -> None:
    with pytest.raises(ValueError, match="enable_prefill_cuda_graph is False"):
        build_generation_batch_overrides(
            max_running_requests=16,
            enable_prefill_cuda_graph=False,
            prefill_graph_token_buckets=[240],
        )


@pytest.mark.parametrize(
    "buckets",
    [[], [240.5], [0], [-1], [True], 240, "240,640"],
)
def test_build_generation_batch_overrides_rejects_invalid_prefill_graph_sizes(
    buckets,
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        build_generation_batch_overrides(
            max_running_requests=16,
            enable_prefill_cuda_graph=True,
            prefill_graph_token_buckets=buckets,
        )
