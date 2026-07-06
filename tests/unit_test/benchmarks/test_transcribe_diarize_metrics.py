# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics.transcribe_diarize_metrics import (
    DiarizationRow,
    _levenshtein_distance,
    clean_no_speaker,
    compute_diarization_metrics,
    split_clean_by_speaker,
)

EVAL_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks/eval/eval_transcribe_diarize.py"
)
LONG_EVAL_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/eval/eval_transcribe_diarize_long.py"
)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_transcribe_diarize_entry",
        EVAL_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_long_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_transcribe_diarize_long_entry",
        LONG_EVAL_SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[S01]我笑了", "我笑了"),
        ("[S01]她喜欢音乐", "她喜欢音乐"),
        ("[S01]I love music", "ilovemusic"),
        ("[S01][笑声]你好", "你好"),
        ("[S01]<silence>Hello [music]", "hello"),
    ],
)
def test_clean_no_speaker_only_strips_marked_events(text: str, expected: str) -> None:
    assert clean_no_speaker(text) == expected


def test_split_clean_by_speaker_preserves_spoken_event_words() -> None:
    assert split_clean_by_speaker(
        "[S01]我笑了[S02]I love music", implicit_single_speaker=False
    ) == {
        "[S1]": "我笑了",
        "[S2]": "ilovemusic",
    }


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("你好世界", "你号世", 2),
        ("[S1]abc", "abc[S2]", 7),
    ],
)
def test_levenshtein_distance_matches_expected(
    reference: str,
    prediction: str,
    expected: int,
) -> None:
    assert _levenshtein_distance(reference, prediction) == expected


def test_compute_diarization_metrics_includes_timestamp_der_for_exact_match() -> None:
    result = compute_diarization_metrics(
        [
            DiarizationRow(
                sample_id="sample-1",
                audio_path="/tmp/sample-1.wav",
                reference_text="[0.00][S01]hello[1.00][1.00][S02]world[2.00]",
                prediction_text="[0.00][S01]hello[1.00][1.00][S02]world[2.00]",
            )
        ]
    )

    assert result.metrics["speaker_timestamp_der"] == pytest.approx(0.0)
    assert result.metrics["speaker_timestamp_der_valid_samples"] == 1
    assert result.metrics["speaker_timestamp_der_skipped"] == 0
    assert result.samples[0].speaker_timestamp_der_valid is True
    assert result.samples[0].speaker_timestamp_der == pytest.approx(0.0)


def test_compute_diarization_metrics_marks_missing_timestamp_prediction_invalid() -> (
    None
):
    result = compute_diarization_metrics(
        [
            DiarizationRow(
                sample_id="sample-1",
                audio_path="/tmp/sample-1.wav",
                reference_text="[0.00][S01]hello[1.00]",
                prediction_text="[S01]hello",
            )
        ]
    )

    assert result.metrics["speaker_timestamp_der"] is None
    assert result.metrics["speaker_timestamp_der_valid_samples"] == 0
    assert result.metrics["speaker_timestamp_der_skipped_no_pred_segments"] == 1
    assert result.samples[0].speaker_timestamp_der_valid is False
    assert (
        result.samples[0].speaker_timestamp_der_invalid_reason
        == "no_pred_timestamped_speaker_segments"
    )


def test_build_metrics_section_prints_timestamp_metrics() -> None:
    module = _load_eval_module()

    section = module._build_metrics_section(
        "diarization_metrics_percent",
        {
            "speaker_timestamp_der": 12.3456,
            "speaker_timestamp_der_valid_samples": 7,
            "speaker_timestamp_der_skipped": 1,
        },
        (
            "speaker_timestamp_der",
            "speaker_timestamp_der_valid_samples",
            "speaker_timestamp_der_skipped",
        ),
    )

    assert "speaker_timestamp_der:" in section
    assert "12.35" in section
    assert "speaker_timestamp_der_valid_samples:" in section


def test_compute_diarization_metrics_partitions_cer_above_50_percent() -> None:
    result = compute_diarization_metrics(
        [
            DiarizationRow(
                sample_id="ok",
                audio_path="/tmp/ok.wav",
                reference_text="[S01]hello",
                prediction_text="[S01]hello",
            ),
            DiarizationRow(
                sample_id="bad",
                audio_path="/tmp/bad.wav",
                reference_text="[S01]abc",
                prediction_text="[S01]" + "d" * 100,
            ),
        ]
    )

    assert result.metrics["cer_no_spk"] is not None
    assert result.metrics["cer_no_spk_below_50_corpus"] == pytest.approx(0.0)
    assert result.metrics["n_above_50_pct_cer"] == 1
    assert result.metrics_percent["cer_no_spk_below_50_corpus"] == pytest.approx(0.0)
    assert result.metrics_percent["n_above_50_pct_cer"] == 1


def test_build_key_metrics_section_prints_partitioned_cer_metrics() -> None:
    module = _load_eval_module()

    section = module._build_key_metrics_section(
        {
            "cer_no_spk": 21.68,
            "cer_no_spk_below_50_corpus": 5.50,
            "n_above_50_pct_cer": 1,
            "cp_cer": 14.42,
            "delta_cer": 7.85,
            "speaker_timestamp_der": 23.97,
        }
    )

    assert "cer_no_spk_below_50_corpus:" in section
    assert "5.5" in section
    assert "n_above_50_pct_cer:" in section


def test_build_key_metrics_section_prints_selected_metrics() -> None:
    module = _load_eval_module()

    section = module._build_key_metrics_section(
        {
            "cer_no_spk": 6.57,
            "cp_cer": 14.42,
            "delta_cer": 7.85,
            "speaker_timestamp_der": 23.97,
        }
    )

    assert "key_metrics" in section
    assert "cer_no_spk:" in section
    assert "6.57" in section
    assert "cp_cer:" in section
    assert "14.42" in section
    assert "delta_cer:" in section
    assert "7.85" in section
    assert "speaker_timestamp_der:" in section
    assert "23.97" in section


def test_extract_prediction_text_prefers_top_level_text_for_timestamps() -> None:
    from benchmarks.tasks.transcribe_diarize import extract_prediction_text

    payload = {
        "text": "[0.00][S01]hello[1.00][1.00][S02]world[2.00]",
        "segments": [
            {"text": "[S01]hello"},
            {"text": "[S02]world"},
        ],
    }

    assert extract_prediction_text(payload) == payload["text"]


def test_long_eval_saves_and_loads_raw_asr_results(tmp_path: Path) -> None:
    module = _load_long_eval_module()

    args = Namespace(
        repo_id="zhaochenyang20/AISHELL4",
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        base_url=None,
        host="127.0.0.1",
        port=8001,
        concurrency=16,
        warmup=0,
        request_rate=float("inf"),
        request_timeout_s=1800,
        max_new_tokens=65536,
        max_samples=20,
        output_dir=str(tmp_path),
        asr_results_file="raw.json",
    )
    samples = [
        module.Movies800Sample(
            sample_id="sample-1",
            audio_path="/tmp/sample-1.wav",
            expected_text="[0.00][S01]hello[1.00]",
        )
    ]
    outputs = [
        RequestResult(
            request_id="sample-1",
            text="[0.00][S01]hello[1.00]",
            is_success=True,
            latency_s=1.2,
            audio_duration_s=4.0,
            rtf=0.3,
        )
    ]

    path = module._save_asr_results(args, samples, outputs, wall_clock_s=1.3)
    loaded_samples, loaded_outputs, loaded_config = module._load_asr_results(path)

    assert loaded_config["request_rate"] == "inf"
    assert loaded_config["max_new_tokens"] == 65536
    assert loaded_samples == samples
    assert loaded_outputs[0].request_id == "sample-1"
    assert loaded_outputs[0].text == "[0.00][S01]hello[1.00]"
    assert loaded_outputs[0].is_success is True
