# SPDX-License-Identifier: Apache-2.0
"""Tests for shared SGLang ServerArgs construction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs  # noqa: E402

from sglang_omni.scheduling.sglang_backend.server_args_builder import (  # noqa: E402
    OmniServerArgs,
    build_sglang_server_args,
)

_MOSS_TD_ARCHITECTURE = "MossTranscribeDiarizeForConditionalGeneration"


@pytest.mark.parametrize(
    (
        "architecture",
        "already_disabled",
        "expected_multimodal",
        "expected_disabled",
    ),
    [
        (_MOSS_TD_ARCHITECTURE, False, False, False),
        ("Qwen3OmniMoeForConditionalGeneration", False, True, True),
        (_MOSS_TD_ARCHITECTURE, True, False, True),
    ],
)
def test_piecewise_prefill_skips_only_the_multimodal_gate(
    architecture: str,
    already_disabled: bool,
    expected_multimodal: bool,
    expected_disabled: bool,
) -> None:
    server_args = OmniServerArgs.__new__(OmniServerArgs)
    server_args.model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(architectures=[architecture]),
    )
    server_args.disable_piecewise_cuda_graph = already_disabled
    observed = []

    def upstream_policy(args: ServerArgs) -> None:
        observed.append(args.model_config.is_multimodal)
        args.disable_piecewise_cuda_graph = (
            args.disable_piecewise_cuda_graph or args.model_config.is_multimodal
        )

    with patch.object(ServerArgs, "_handle_piecewise_cuda_graph", upstream_policy):
        server_args._handle_piecewise_cuda_graph()

    assert observed == [expected_multimodal]
    assert server_args.disable_piecewise_cuda_graph is expected_disabled
    assert server_args.model_config.is_multimodal is True


def test_builder_uses_shared_omni_server_args() -> None:
    server_args = build_sglang_server_args(
        "dummy",
        context_length=1024,
    )

    assert isinstance(server_args, OmniServerArgs)
