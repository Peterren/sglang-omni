# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.benchmarker import utils as benchmark_utils


def test_managed_omni_server_reclaims_gpu_on_startup_failure(monkeypatch) -> None:
    cleanup_calls: list[str] = []

    def _fail_start(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("startup failed")

    monkeypatch.setattr(benchmark_utils, "_ensure_port_available", lambda *args: None)
    monkeypatch.setattr(benchmark_utils, "start_server_from_cmd", _fail_start)
    monkeypatch.setattr(
        benchmark_utils,
        "wait_for_gpu_memory_release",
        lambda: cleanup_calls.append("cleanup"),
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        with benchmark_utils.managed_omni_server(
            model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
            port=18080,
            host="127.0.0.1",
            log_file=Path("/tmp/test-asr-startup.log"),
        ):
            pass

    assert cleanup_calls == ["cleanup"]


def test_managed_omni_server_can_skip_startup_failure_gpu_cleanup(
    monkeypatch,
) -> None:
    cleanup_calls: list[str] = []

    def _fail_start(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("startup failed")

    monkeypatch.setattr(benchmark_utils, "_ensure_port_available", lambda *args: None)
    monkeypatch.setattr(benchmark_utils, "start_server_from_cmd", _fail_start)
    monkeypatch.setattr(
        benchmark_utils,
        "wait_for_gpu_memory_release",
        lambda: cleanup_calls.append("cleanup"),
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        with benchmark_utils.managed_omni_server(
            model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
            port=18080,
            host="127.0.0.1",
            log_file=None,
            wait_for_gpu_release=False,
        ):
            pass

    assert cleanup_calls == []
