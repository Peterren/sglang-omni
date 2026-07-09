from __future__ import annotations

import asyncio
import json
import os

import pytest

from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_results_config,
    _config_from_args,
    _write_request_profile_report,
    run_tts_seedtts_benchmark,
)


def _config_from_cli(*args: str) -> TtsSeedttsBenchmarkConfig:
    parser = _build_arg_parser()
    return _config_from_args(parser.parse_args(list(args)))


def test_seedtts_benchmark_batch_args_default_to_64() -> None:
    config = _config_from_cli()

    assert config.max_running_requests == 64
    assert config.cuda_graph_max_bs == 64

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 64
    assert results_config["cuda_graph_max_bs"] == 64


def test_seedtts_benchmark_batch_args_are_independent() -> None:
    config = _config_from_cli(
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "128",
    )

    assert config.max_running_requests == 32
    assert config.cuda_graph_max_bs == 128

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 32
    assert results_config["cuda_graph_max_bs"] == 128


def test_seedtts_benchmark_profile_args_are_recorded() -> None:
    config = _config_from_cli(
        "--profile-request-events",
        "--warmup",
        "0",
        "--profile-run-id",
        "reference-encode-gate",
        "--profile-event-dir",
        "/tmp/events",
        "--profile-report-path",
        "/tmp/report.json",
        "--require-reference-encode-profile",
    )

    assert config.profile_request_events is True
    assert config.profile_run_id == "reference-encode-gate"
    assert config.profile_event_dir == "/tmp/events"
    assert config.profile_report_path == "/tmp/report.json"
    assert config.require_reference_encode_profile is True
    assert config.warmup == 0

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["profile_request_events"] is True
    assert results_config["profile_run_id"] == "reference-encode-gate"
    assert results_config["profile_event_dir"] == "/tmp/events"
    assert results_config["profile_report_path"] == "/tmp/report.json"
    assert results_config["require_reference_encode_profile"] is True


def test_seedtts_profile_run_requires_zero_warmup() -> None:
    config = TtsSeedttsBenchmarkConfig(
        model="fishaudio/s2-pro",
        meta="zhaochenyang20/seed-tts-eval-arrow",
        profile_request_events=True,
        warmup=1,
    )

    with pytest.raises(ValueError, match="warmup=0"):
        asyncio.run(run_tts_seedtts_benchmark(config))


def test_seedtts_profile_report_fails_when_events_are_missing(tmp_path) -> None:
    report_path = tmp_path / "report.json"

    with pytest.raises(RuntimeError, match="No request profile events"):
        _write_request_profile_report(
            event_dir=str(tmp_path / "missing-events"),
            report_path=str(report_path),
            expect_events=True,
        )

    assert not report_path.exists()


def test_seedtts_profile_report_can_require_reference_encode_events(tmp_path) -> None:
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    event = {
        "request_id": "r1",
        "stage": "coordinator",
        "event_name": "request_admission",
        "timestamp_ns": 0,
        "run_id": "run",
        "pid": os.getpid(),
        "metadata": {},
    }
    (event_dir / "events_coordinator_1.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No reference encode profile events"):
        _write_request_profile_report(
            event_dir=str(event_dir),
            report_path=str(tmp_path / "report.json"),
            expect_events=True,
            expect_reference_encode=True,
        )


def test_seedtts_profile_report_can_require_successful_reference_encode(
    tmp_path,
) -> None:
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    event = {
        "request_id": "r1",
        "stage": "preprocessing",
        "event_name": "reference_encode_lookup",
        "timestamp_ns": 0,
        "run_id": "run",
        "pid": os.getpid(),
        "metadata": {
            "model_id": "fish",
            "encoder_id": "codec",
            "artifact_kind": "codes",
            "result": "hit",
        },
    }
    (event_dir / "events_preprocessing_1.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No successful reference encode misses"):
        _write_request_profile_report(
            event_dir=str(event_dir),
            report_path=str(tmp_path / "report.json"),
            expect_events=True,
            expect_reference_encode=True,
        )


def test_seedtts_profile_report_rejects_uncacheable_only_reference_encode(
    tmp_path,
) -> None:
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    events = [
        {
            "request_id": "r1",
            "stage": "preprocessing",
            "event_name": "reference_encode_lookup",
            "timestamp_ns": 0,
            "run_id": "run",
            "pid": os.getpid(),
            "metadata": {"result": "uncacheable"},
        },
        {
            "request_id": "r1",
            "stage": "preprocessing",
            "event_name": "reference_encode_start",
            "timestamp_ns": 1_000,
            "run_id": "run",
            "pid": os.getpid(),
            "metadata": {"result": "uncacheable"},
        },
        {
            "request_id": "r1",
            "stage": "preprocessing",
            "event_name": "reference_encode_end",
            "timestamp_ns": 2_000,
            "run_id": "run",
            "pid": os.getpid(),
            "metadata": {"result": "success"},
        },
    ]
    (event_dir / "events_preprocessing_1.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No successful reference encode misses"):
        _write_request_profile_report(
            event_dir=str(event_dir),
            report_path=str(tmp_path / "report.json"),
            expect_events=True,
            expect_reference_encode=True,
        )
