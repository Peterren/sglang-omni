# SPDX-License-Identifier: Apache-2.0
"""Run the reproducible Audar-TTS refactor comparison workload."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
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
REFERENCE_FILE = "samples/demo_male_1_ar.wav"
REFERENCE_TEXT = (
    "لا يمكنني الانتظار لأخبرك — [excited] لقد أنجزنا المشروع أخيراً بعد كلّ "
    "هذا التعب، [laughs] وصدّقني، إنه أجمل شعورٍ على الإطلاق!"
)
TARGET_TEXT = "يسرّنا أن نختبر اليوم نظام تحويل النص العربي إلى كلام واضح وطبيعي."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="audarai/Audar-TTS-V1-Turbo")
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


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
    request_id: str,
    reference_path: Path,
    *,
    seed: int,
    max_new_tokens: int,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={
                "text": TARGET_TEXT,
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
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = np.rint(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.reference_path
    if reference_path is None:
        reference_path = Path(
            hf_hub_download(
                "audarai/Audar-TTS-V1-Turbo",
                REFERENCE_FILE,
                revision=AUDAR_REVISION,
            )
        )

    _sync()
    init_started = time.perf_counter()
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
    _sync()
    initialization_s = time.perf_counter() - init_started

    iterations: list[dict[str, Any]] = []
    for index in range(args.repeats):
        payload = _payload(
            f"{args.label}-{index}",
            reference_path,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        payload, preprocessing_s = _timed(lambda: preprocessing._fn(payload))
        payload, reference_s = _timed(lambda: reference_encoder._fn(payload))
        payload, engine_wall_s = _timed(lambda: tts_engine._fn(payload))
        generated_state = AudarTTSState.from_dict(payload.data)
        codes = np.asarray(generated_state.audio_codes, dtype=np.int32)
        payload, vocoder_s = _timed(lambda: asyncio.run(vocoder._fn(payload)))
        waveform = np.frombuffer(payload.data["audio_waveform"], dtype=np.float32)
        sample_rate = int(payload.data["sample_rate"])
        output_duration_s = waveform.size / sample_rate
        total_s = preprocessing_s + reference_s + engine_wall_s + vocoder_s
        if not np.isfinite(waveform).all() or waveform.size == 0:
            raise RuntimeError("Audar-TTS emitted an invalid waveform")

        iteration = {
            "index": index,
            "preprocessing_s": preprocessing_s,
            "reference_s": reference_s,
            "engine_wall_s": engine_wall_s,
            "engine_reported_s": generated_state.engine_time_s,
            "vocoder_s": vocoder_s,
            "total_s": total_s,
            "output_duration_s": output_duration_s,
            "rtf": total_s / output_duration_s,
            "prompt_tokens": generated_state.prompt_tokens,
            "completion_tokens": generated_state.completion_tokens,
            "engine_tokens_per_s": (
                generated_state.completion_tokens / generated_state.engine_time_s
            ),
            "audio_code_count": int(codes.size),
            "audio_code_sha256": _sha256(codes.tobytes()),
            "waveform_sample_count": int(waveform.size),
            "waveform_sha256": _sha256(waveform.tobytes()),
            "waveform_rms": float(np.sqrt(np.mean(np.square(waveform)))),
            "waveform_peak": float(np.max(np.abs(waveform))),
        }
        if not all(
            math.isfinite(value)
            for key, value in iteration.items()
            if key.endswith("_s") or key in {"rtf", "engine_tokens_per_s"}
        ):
            raise RuntimeError("Audar-TTS emitted non-finite benchmark metrics")
        iterations.append(iteration)
        if index == 0:
            _write_wav(
                args.output_dir / f"{args.label}.wav",
                waveform,
                sample_rate,
            )

    code_hashes = {item["audio_code_sha256"] for item in iterations}
    waveform_hashes = {item["waveform_sha256"] for item in iterations}
    result = {
        "label": args.label,
        "audar_revision": AUDAR_REVISION,
        "codec_revision": CODEC_REVISION,
        "model_path": args.model_path,
        "reference_file": REFERENCE_FILE,
        "reference_text": REFERENCE_TEXT,
        "target_text": TARGET_TEXT,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "initialization_s": initialization_s,
        "deterministic_codes": len(code_hashes) == 1,
        "deterministic_waveform": len(waveform_hashes) == 1,
        "summary": {
            key: _summarize([float(item[key]) for item in iterations])
            for key in (
                "reference_s",
                "engine_wall_s",
                "vocoder_s",
                "total_s",
                "rtf",
                "engine_tokens_per_s",
            )
        },
        "iterations": iterations,
    }
    output_path = args.output_dir / f"{args.label}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
