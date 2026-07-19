# SPDX-License-Identifier: Apache-2.0
"""Materialize the selected FLEURS source audio for an Arabic ASR baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from benchmarks.audar_tts.run_quality_benchmark import FLEURS_REVISION, _load_targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo", default="google/fleurs")
    parser.add_argument("--dataset-config", default="ar_eg")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-revision", default=FLEURS_REVISION)
    parser.add_argument("--text-column", default="transcription")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=20)
    parser.add_argument("--include-digits", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(args)

    dataset_file = f"{args.dataset_config}/{args.dataset_split}-00000-of-00001.parquet"
    parquet_path = hf_hub_download(
        args.dataset_repo,
        dataset_file,
        revision=args.dataset_revision,
        repo_type="dataset",
    )
    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )

    samples: list[dict] = []
    generated: list[dict] = []
    for target in targets:
        decoded = dataset[target["dataset_row_index"]]["audio"].get_all_samples()
        waveform = decoded.data.detach().cpu().float().numpy()
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        waveform = np.asarray(waveform, dtype=np.float32)
        sample_rate = int(decoded.sample_rate)
        wav_path = audio_dir / f"{target['sample_id']}.wav"
        sf.write(wav_path, waveform, sample_rate, subtype="PCM_16", format="WAV")
        duration_s = waveform.size / sample_rate
        wav_sha256 = _sha256(wav_path)
        samples.append(
            {
                **target,
                "is_success": True,
                "reached_max_new_tokens": False,
                "wav_sha256": wav_sha256,
                "sample_rate": sample_rate,
                "audio_duration_s": duration_s,
            }
        )
        generated.append(
            {
                "sample_id": target["sample_id"],
                "target_text": target["target_text"],
                "wav_path": str(wav_path.resolve()),
                "is_success": True,
                "latency_s": 0.0,
                "audio_duration_s": duration_s,
            }
        )

    dataset_metadata = {
        "repo": args.dataset_repo,
        "config": args.dataset_config,
        "split": args.dataset_split,
        "revision": args.dataset_revision,
        "file": dataset_file,
        "selection": {
            "samples": args.samples,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "minimum_arabic_letter_fraction": 0.9,
            "exclude_digits": not args.include_digits,
            "text_column": args.text_column,
        },
    }
    result = {
        "schema_version": 1,
        "label": "fleurs-reference-audio",
        "dataset": dataset_metadata,
        "successful_samples": len(samples),
        "truncated_samples": 0,
        "samples": samples,
    }
    (args.output_dir / "reference_audio_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "generated.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
