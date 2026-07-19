from __future__ import annotations

import pytest

from benchmarks.audar_tts.summarize_quality import (
    _compare_generations,
    _quality_metrics,
)


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
                "target_text": "مرحبا بالعالم",
                "is_success": True,
                "reached_max_new_tokens": False,
            },
            {
                "sample_id": "two",
                "target_text": "هذا اختبار بسيط",
                "is_success": True,
                "reached_max_new_tokens": False,
            },
        ]
    )
    result = _quality_metrics(
        {
            "config": {"asr_model": "test-asr"},
            "summary": {"wer_corpus": 0.2},
            "per_sample": [
                {
                    "id": "one",
                    "is_success": True,
                    "ref_norm": "مرحبا بالعالم",
                    "hyp_norm": "مرحبا بالعالم",
                },
                {
                    "id": "two",
                    "is_success": True,
                    "ref_norm": "هذا اختبار بسيط",
                    "hyp_norm": "هذا اختبار",
                },
            ],
        },
        generation,
    )

    assert result["sample_count"] == 2
    assert result["asr_model"] == "test-asr"
    assert result["arabic_wer"] == 0.2
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
                "ref_norm": "مرجع خاطئ",
                "hyp_norm": "مرجع خاطئ",
            },
            {
                "id": "two",
                "is_success": True,
                "ref_norm": "النص الثاني",
                "hyp_norm": "النص الثاني",
            },
        ],
    }

    with pytest.raises(ValueError, match="ASR reference does not match target text"):
        _quality_metrics(wer, generation)
