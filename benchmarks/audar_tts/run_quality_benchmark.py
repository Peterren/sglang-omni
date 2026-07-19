# SPDX-License-Identifier: Apache-2.0
"""Generate a fixed Arabic text set for direct ASR quality evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from sglang_omni.models.audar_tts import stages
from sglang_omni.models.audar_tts.payload_types import AudarTTSState
from sglang_omni.proto import OmniRequest, StagePayload

AUDAR_REVISION = "51f5635f32de3ab45ff28a4b958464532225b247"
CODEC_REVISION = "30c1fdd19e68aee65d542cf043750d4c0165893e"
FLEURS_REVISION = "ab93cf03f9d0cd083c853fad065a6377067408aa"
REFERENCE_FILE = "samples/demo_male_1_ar.wav"
REFERENCE_TEXT = (
    "لا يمكنني الانتظار لأخبرك — [excited] لقد أنجزنا المشروع أخيراً بعد كلّ "
    "هذا التعب، [laughs] وصدّقني، إنه أجمل شعورٍ على الإطلاق!"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="audarai/Audar-TTS-V1-Turbo")
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--dataset-repo", default="google/fleurs")
    parser.add_argument("--dataset-config", default="ar_eg")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-revision", default=FLEURS_REVISION)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=20)
    parser.add_argument("--include-digits", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def _git_commit(checkout: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_arabic_text(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    arabic_letters = [
        char for char in letters if "ARABIC" in unicodedata.name(char, "")
    ]
    return len(arabic_letters) / len(letters) >= 0.9


def _load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

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
    text_column = next(
        (
            column
            for column in ("transcription", "raw_transcription", "text")
            if column in dataset.column_names
        ),
        None,
    )
    if text_column is None:
        raise ValueError(f"no text column in dataset: {dataset.column_names}")
    id_column = "id" if "id" in dataset.column_names else None
    retained_columns = [text_column] + ([id_column] if id_column else [])
    dataset = dataset.select_columns(retained_columns)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, row in enumerate(dataset):
        text = " ".join(str(row[text_column]).split())
        word_count = len(text.split())
        if not args.min_words <= word_count <= args.max_words:
            continue
        if not args.include_digits and any(char.isdigit() for char in text):
            continue
        if not _is_arabic_text(text) or text in seen:
            continue
        seen.add(text)
        selected.append(
            {
                "sample_id": f"fleurs-ar-eg-{row_index:04d}",
                "dataset_row_index": row_index,
                "dataset_id": row[id_column] if id_column else row_index,
                "target_text": text,
                "word_count": word_count,
            }
        )
        if len(selected) == args.samples:
            break
    if len(selected) != args.samples:
        raise ValueError(f"selected {len(selected)} of {args.samples} requested texts")
    return selected


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call: Any) -> tuple[Any, float]:
    _sync()
    started = time.perf_counter()
    result = call()
    _sync()
    return result, time.perf_counter() - started


def _payload(
    *,
    request_id: str,
    target_text: str,
    reference_path: Path,
    seed: int,
    max_new_tokens: int,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={
                "text": target_text,
                "references": [
                    {
                        "audio_path": str(reference_path),
                        "text": REFERENCE_TEXT,
                    }
                ],
            },
            params={"max_new_tokens": max_new_tokens},
            metadata={"tts_params": {"seed": seed}},
        ),
        data={},
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def main() -> None:
    args = _parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(args)

    reference_path = args.reference_path
    if reference_path is None:
        reference_path = Path(
            hf_hub_download(
                "audarai/Audar-TTS-V1-Turbo",
                REFERENCE_FILE,
                revision=AUDAR_REVISION,
            )
        )

    preprocessing = stages.create_preprocessing_executor()
    reference_encoder = stages.create_reference_encoder_executor(
        gpu_id=0,
        codec_revision=CODEC_REVISION,
    )
    tts_engine = stages.create_tts_engine_executor(
        args.model_path,
        gpu_id=0,
        model_revision=AUDAR_REVISION,
    )
    vocoder = stages.create_vocoder_executor(
        gpu_id=0,
        codec_revision=CODEC_REVISION,
    )

    samples: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for target in targets:
        started = time.perf_counter()
        sample = {**target, "is_success": False, "error": None}
        generated_entry = {
            "sample_id": target["sample_id"],
            "target_text": target["target_text"],
            "wav_path": "",
            "is_success": False,
            "latency_s": 0.0,
            "audio_duration_s": 0.0,
        }
        try:
            payload = _payload(
                request_id=f"{args.label}-{target['sample_id']}",
                target_text=target["target_text"],
                reference_path=reference_path,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            payload, preprocessing_s = _timed(lambda: preprocessing._fn(payload))
            payload, reference_s = _timed(lambda: reference_encoder._fn(payload))
            payload, engine_s = _timed(lambda: tts_engine._fn(payload))
            state = AudarTTSState.from_dict(payload.data)
            codes = np.asarray(state.audio_codes, dtype=np.int32)
            payload, vocoder_s = _timed(lambda: asyncio.run(vocoder._fn(payload)))
            waveform = np.frombuffer(payload.data["audio_waveform"], dtype=np.float32)
            sample_rate = int(payload.data["sample_rate"])
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise RuntimeError("invalid waveform")
            wav_path = audio_dir / f"{target['sample_id']}.wav"
            _write_wav(wav_path, waveform, sample_rate)
            duration_s = waveform.size / sample_rate
            sample.update(
                {
                    "is_success": True,
                    "audio_code_count": int(codes.size),
                    "audio_code_sha256": _sha256(codes.tobytes()),
                    "waveform_sample_count": int(waveform.size),
                    "waveform_sha256": _sha256(waveform.tobytes()),
                    "wav_sha256": _sha256(wav_path.read_bytes()),
                    "sample_rate": sample_rate,
                    "audio_duration_s": duration_s,
                    "completion_tokens": state.completion_tokens,
                    "reached_max_new_tokens": (
                        state.completion_tokens >= args.max_new_tokens
                    ),
                    "preprocessing_s": preprocessing_s,
                    "reference_s": reference_s,
                    "engine_s": engine_s,
                    "vocoder_s": vocoder_s,
                    "total_s": preprocessing_s + reference_s + engine_s + vocoder_s,
                }
            )
            generated_entry.update(
                {
                    "wav_path": str(wav_path.resolve()),
                    "is_success": True,
                    "latency_s": sample["total_s"],
                    "audio_duration_s": duration_s,
                }
            )
        except Exception as exc:
            sample["error"] = str(exc)
            generated_entry["error"] = str(exc)
            generated_entry["latency_s"] = time.perf_counter() - started
        samples.append(sample)
        generated.append(generated_entry)

    result = {
        "schema_version": 1,
        "label": args.label,
        "commit": _git_commit(args.checkout.resolve()),
        "model_path": args.model_path,
        "audar_revision": AUDAR_REVISION,
        "codec_revision": CODEC_REVISION,
        "reference_file": REFERENCE_FILE,
        "dataset": {
            "repo": args.dataset_repo,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "revision": args.dataset_revision,
            "file": (
                f"{args.dataset_config}/{args.dataset_split}-00000-of-00001.parquet"
            ),
            "selection": {
                "samples": args.samples,
                "min_words": args.min_words,
                "max_words": args.max_words,
                "minimum_arabic_letter_fraction": 0.9,
                "exclude_digits": not args.include_digits,
            },
        },
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "successful_samples": sum(sample["is_success"] for sample in samples),
        "truncated_samples": sum(
            sample.get("reached_max_new_tokens", False) for sample in samples
        ),
        "samples": samples,
    }
    (args.output_dir / "generation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "generated.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["successful_samples"] != args.samples or result["truncated_samples"]:
        raise RuntimeError(
            "quality generation had failed or max-token-truncated samples"
        )


if __name__ == "__main__":
    main()
