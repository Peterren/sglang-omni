from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.audar_tts.summarize_quality import (
    _compare_generations,
    _quality_metrics,
)
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.tts import save_wer_results


def _generation_result(samples: list[dict]) -> dict:
    return {
        "dataset": {"revision": "test"},
        "successful_samples": len(samples),
        "truncated_samples": 0,
        "samples": samples,
    }


def test_arabic_quality_uses_target_and_asr_text_directly() -> None:
    generation = _generation_result(
        [
            {
                "sample_id": "one",
                "target_text": "مرحبا بكم في هذا العالم الجميل",
                "is_success": True,
                "reached_max_new_tokens": False,
                "wav_sha256": "wav-one",
            },
            {
                "sample_id": "two",
                "target_text": "هذا اختبار عربي بسيط وواضح جدا",
                "is_success": True,
                "reached_max_new_tokens": False,
                "wav_sha256": "wav-two",
            },
        ]
    )
    result = _quality_metrics(
        {
            "config": {"asr_model": "test-asr"},
            "summary": {"wer_corpus": 1 / 12},
            "per_sample": [
                {
                    "id": "one",
                    "is_success": True,
                    "wav_sha256": "wav-one",
                    "ref_norm": "مرحبا بكم في هذا العالم الجميل",
                    "hyp_norm": "مرحبا بكم في هذا العالم الجميل",
                },
                {
                    "id": "two",
                    "is_success": True,
                    "wav_sha256": "wav-two",
                    "ref_norm": "هذا اختبار عربي بسيط وواضح جدا",
                    "hyp_norm": "هذا اختبار عربي بسيط جدا",
                },
            ],
        },
        generation,
    )

    assert result["sample_count"] == 2
    assert result["asr_model"] == "test-asr"
    assert result["arabic_wer"] == pytest.approx(1 / 12)
    assert result["samples_above_50_percent_wer"] == 0
    assert 0 < result["arabic_cer"] < 1
    assert 0 < result["arabic_bleu"] < 100
    assert 0 < result["arabic_chrf_pp"] < 100


def test_paired_quality_requires_every_hash_to_match() -> None:
    samples = [
        {
            "sample_id": "one",
            "target_text": "النص الأول",
            "is_success": True,
            "reached_max_new_tokens": False,
            "audio_code_sha256": "codes",
            "waveform_sha256": "waveform",
            "wav_sha256": "wav",
        },
        {
            "sample_id": "two",
            "target_text": "النص الثاني",
            "is_success": True,
            "reached_max_new_tokens": False,
            "audio_code_sha256": "codes-2",
            "waveform_sha256": "waveform-2",
            "wav_sha256": "wav-2",
        },
    ]

    exactness = _compare_generations(
        _generation_result(samples), _generation_result(samples)
    )

    assert exactness == {
        "sample_count": 2,
        "audio_code_hash_matches": 2,
        "waveform_hash_matches": 2,
        "wav_hash_matches": 2,
        "all_outputs_exact": True,
    }


def test_arabic_quality_rejects_asr_reference_not_derived_from_target() -> None:
    generation = _generation_result(
        [
            {
                "sample_id": sample_id,
                "target_text": target_text,
                "is_success": True,
                "reached_max_new_tokens": False,
                "wav_sha256": f"wav-{sample_id}",
            }
            for sample_id, target_text in (
                ("one", "النص الأول"),
                ("two", "النص الثاني"),
            )
        ]
    )
    wer = {
        "config": {"asr_model": "test-asr"},
        "summary": {"wer_corpus": 0.0},
        "per_sample": [
            {
                "id": "one",
                "is_success": True,
                "wav_sha256": "wav-one",
                "ref_norm": "مرجع خاطئ",
                "hyp_norm": "مرجع خاطئ",
            },
            {
                "id": "two",
                "is_success": True,
                "wav_sha256": "wav-two",
                "ref_norm": "النص الثاني",
                "hyp_norm": "النص الثاني",
            },
        ],
    }

    with pytest.raises(ValueError, match="ASR reference does not match target text"):
        _quality_metrics(wer, generation)


def test_arabic_quality_rejects_asr_audio_not_from_generation() -> None:
    generation = _generation_result(
        [
            {
                "sample_id": sample_id,
                "target_text": target_text,
                "is_success": True,
                "reached_max_new_tokens": False,
                "wav_sha256": f"wav-{sample_id}",
            }
            for sample_id, target_text in (
                ("one", "النص الأول"),
                ("two", "النص الثاني"),
            )
        ]
    )
    wer = {
        "config": {"asr_model": "test-asr"},
        "summary": {"wer_corpus": 0.0},
        "per_sample": [
            {
                "id": "one",
                "is_success": True,
                "wav_sha256": "different-wav",
                "ref_norm": "النص الاول",
                "hyp_norm": "النص الاول",
            },
            {
                "id": "two",
                "is_success": True,
                "wav_sha256": "wav-two",
                "ref_norm": "النص الثاني",
                "hyp_norm": "النص الثاني",
            },
        ],
    }

    with pytest.raises(ValueError, match="ASR WAV does not match generated WAV"):
        _quality_metrics(wer, generation)


def test_wer_artifact_records_transcribed_wav_hash(tmp_path: Path) -> None:
    output = SampleOutput(
        sample_id="one",
        target_text="النص الأول",
        whisper_text="النص الأول",
        wav_sha256="abc123",
        is_success=True,
    )

    save_wer_results([output], {}, {}, str(tmp_path))

    artifact = json.loads((tmp_path / "wer_results.json").read_text())
    assert artifact["per_sample"][0]["wav_sha256"] == "abc123"
