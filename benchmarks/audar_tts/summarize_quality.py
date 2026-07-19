# SPDX-License-Identifier: Apache-2.0
"""Summarize direct Arabic ASR metrics for paired Audar generations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from jiwer import process_characters, process_words
from sacrebleu.metrics import BLEU, CHRF

from benchmarks.tasks.asr import normalize_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-t1-generation", type=Path, required=True)
    parser.add_argument("--latest-generation", type=Path, required=True)
    parser.add_argument("--latest-wer", type=Path, required=True)
    parser.add_argument("--pre-t1-wer", type=Path)
    parser.add_argument("--reference-audio-generation", type=Path, required=True)
    parser.add_argument("--reference-wer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_generation_samples(
    result: dict[str, Any], *, label: str
) -> dict[str, dict]:
    samples = result["samples"]
    if result["successful_samples"] != len(samples) or any(
        not sample["is_success"] for sample in samples
    ):
        raise ValueError(f"{label} generation contains failed samples")
    if result["truncated_samples"] or any(
        sample["reached_max_new_tokens"] for sample in samples
    ):
        raise ValueError(f"{label} generation contains truncated samples")
    by_id = {sample["sample_id"]: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError(f"{label} generation contains duplicate sample IDs")
    return by_id


def _compare_generations(
    pre_t1: dict[str, Any], latest: dict[str, Any]
) -> dict[str, Any]:
    if pre_t1["dataset"] != latest["dataset"]:
        raise ValueError("pre-T1 and latest dataset metadata differ")
    pre_samples = _complete_generation_samples(pre_t1, label="pre-T1")
    latest_samples = _complete_generation_samples(latest, label="latest")
    if pre_samples.keys() != latest_samples.keys():
        raise ValueError("pre-T1 and latest sample IDs differ")
    sample_ids = sorted(pre_samples)
    if any(
        pre_samples[sample_id]["target_text"]
        != latest_samples[sample_id]["target_text"]
        for sample_id in sample_ids
    ):
        raise ValueError("pre-T1 and latest target texts differ")
    comparisons = {
        field: sum(
            pre_samples[sample_id][field] == latest_samples[sample_id][field]
            for sample_id in sample_ids
        )
        for field in (
            "audio_code_sha256",
            "waveform_sha256",
            "wav_sha256",
        )
    }
    return {
        "sample_count": len(sample_ids),
        "audio_code_hash_matches": comparisons["audio_code_sha256"],
        "waveform_hash_matches": comparisons["waveform_sha256"],
        "wav_hash_matches": comparisons["wav_sha256"],
        "all_outputs_exact": all(
            count == len(sample_ids) for count in comparisons.values()
        ),
    }


def _quality_metrics(wer: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    generation_samples = _complete_generation_samples(generation, label="quality")
    samples = [sample for sample in wer["per_sample"] if sample["is_success"]]
    samples_by_id = {sample["id"]: sample for sample in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("ASR result contains duplicate sample IDs")
    if samples_by_id.keys() != generation_samples.keys():
        raise ValueError("ASR and generation sample IDs differ")
    if len(samples_by_id) < 2:
        raise ValueError("Arabic quality evaluation requires at least two samples")

    references: list[str] = []
    hypotheses: list[str] = []
    for sample_id in sorted(samples_by_id):
        expected_reference = normalize_text(
            generation_samples[sample_id]["target_text"], "ar"
        )
        sample = samples_by_id[sample_id]
        if sample["ref_norm"] != expected_reference:
            raise ValueError(f"ASR reference does not match target text: {sample_id}")
        if sample.get("wav_sha256") != generation_samples[sample_id]["wav_sha256"]:
            raise ValueError(f"ASR WAV does not match generated WAV: {sample_id}")
        references.append(expected_reference)
        hypotheses.append(sample["hyp_norm"])

    word_measures = process_words(references, hypotheses)
    recorded_wer = float(wer["summary"]["wer_corpus"])
    if not math.isclose(word_measures.wer, recorded_wer, rel_tol=1e-12):
        raise ValueError("recorded corpus WER does not match direct recomputation")
    per_sample_wers = [
        process_words(reference, hypothesis).wer
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    below_50 = [
        (reference, hypothesis)
        for reference, hypothesis, sample_wer in zip(
            references, hypotheses, per_sample_wers, strict=True
        )
        if sample_wer <= 0.5
    ]
    below_50_wer = process_words(
        [reference for reference, _ in below_50],
        [hypothesis for _, hypothesis in below_50],
    ).wer

    cer_errors = 0
    cer_reference_characters = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        measures = process_characters(
            reference.replace(" ", ""), hypothesis.replace(" ", "")
        )
        cer_errors += measures.substitutions + measures.deletions + measures.insertions
        cer_reference_characters += (
            measures.substitutions + measures.deletions + measures.hits
        )
    bleu = BLEU(tokenize="none")
    bleu_score = bleu.corpus_score(hypotheses, [references])
    return {
        "sample_count": len(samples_by_id),
        "asr_model": wer["config"]["asr_model"],
        "arabic_wer": word_measures.wer,
        "arabic_wer_excluding_above_50_percent": below_50_wer,
        "samples_above_50_percent_wer": sum(
            sample_wer > 0.5 for sample_wer in per_sample_wers
        ),
        "max_sample_wer": max(per_sample_wers),
        "arabic_cer": cer_errors / cer_reference_characters,
        "arabic_bleu": bleu_score.score,
        "arabic_bleu_signature": str(bleu.get_signature()),
        "arabic_chrf_pp": CHRF(word_order=2)
        .corpus_score(hypotheses, [references])
        .score,
        "normalization": "NFKC, Arabic diacritics/tatweel/punctuation stripped, Alef and digits normalized",
    }


def main() -> None:
    args = _parse_args()
    pre_generation = _load(args.pre_t1_generation)
    latest_generation = _load(args.latest_generation)
    reference_generation = _load(args.reference_audio_generation)
    if reference_generation["dataset"] != latest_generation["dataset"]:
        raise ValueError("reference-audio and TTS dataset metadata differ")
    reference_samples = _complete_generation_samples(
        reference_generation, label="reference audio"
    )
    latest_samples = _complete_generation_samples(latest_generation, label="latest")
    if reference_samples.keys() != latest_samples.keys() or any(
        reference_samples[sample_id]["target_text"]
        != latest_samples[sample_id]["target_text"]
        for sample_id in reference_samples
    ):
        raise ValueError("reference-audio and TTS target corpora differ")
    exactness = _compare_generations(pre_generation, latest_generation)
    latest_quality = _quality_metrics(_load(args.latest_wer), latest_generation)
    reference_quality = _quality_metrics(
        _load(args.reference_wer), reference_generation
    )
    if exactness["all_outputs_exact"]:
        pre_quality = dict(latest_quality)
        pre_quality["evidence"] = "exact waveform match with latest"
    else:
        if args.pre_t1_wer is None:
            raise ValueError("--pre-t1-wer is required when waveforms differ")
        pre_quality = _quality_metrics(_load(args.pre_t1_wer), pre_generation)
        pre_quality["evidence"] = "independent ASR evaluation"
    latest_quality["evidence"] = "direct ASR evaluation"
    summary = {
        "schema_version": 3,
        "dataset": latest_generation["dataset"],
        "commits": {
            "pre_t1": pre_generation["commit"],
            "latest": latest_generation["commit"],
        },
        "exactness": exactness,
        "pre_t1": pre_quality,
        "latest": latest_quality,
        "reference_audio_asr_baseline": reference_quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
