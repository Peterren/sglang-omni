# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Audar-TTS-V1 Turbo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.audar_tts.payload_types import AudarTTSState
from sglang_omni.models.audar_tts.protocol import (
    TARGET_CODES_END,
    build_prompt,
    parse_speech_codes,
)
from sglang_omni.models.audar_tts.request_builders import build_audar_state
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

DEFAULT_GGUF_FILENAME = "Audar-TTS-V1-Turbo-Q4_K_M.gguf"
DEFAULT_CODEC_MODEL = "neuphonic/neucodec"
REFERENCE_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
MIN_REFERENCE_SECONDS = 5.0
MAX_REFERENCE_SECONDS = 15.0


@dataclass(frozen=True)
class _ReferenceInput:
    source_kind: str
    source: Any
    media_type: str | None = None


@lru_cache(maxsize=None)
def _load_codec(model: str, revision: str, device: str) -> Any:
    try:
        from neucodec import NeuCodec
    except ImportError as exc:
        raise RuntimeError(
            "Audar-TTS requires the 'audar-tts' optional dependencies"
        ) from exc
    return NeuCodec.from_pretrained(model, revision=revision).eval().to(device)


def _device(gpu_id: int | None) -> str:
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


def _normalize_reference(raw_input: Any) -> _ReferenceInput:
    if not isinstance(raw_input, dict):
        raise TypeError("Audar-TTS reference input must be a dict")
    if raw_input.get("audio_path") is not None:
        return _ReferenceInput("path", str(raw_input["audio_path"]))
    if raw_input.get("bytes") is not None:
        return _ReferenceInput("bytes", bytes(raw_input["bytes"]))
    data = raw_input.get("base64") or raw_input.get("data")
    if data is not None:
        return _ReferenceInput(
            "base64", str(data), str(raw_input.get("media_type") or "audio/wav")
        )
    raise ValueError("Audar-TTS reference input has no audio payload")


def _load_reference_waveform(item: _ReferenceInput) -> torch.Tensor:
    audio_io = AudioMediaIO(target_sr=REFERENCE_SAMPLE_RATE)
    if item.source_kind == "path":
        audio, _ = audio_io.load_file(Path(item.source).expanduser())
    elif item.source_kind == "bytes":
        audio, _ = audio_io.load_bytes(item.source)
    elif item.source_kind == "base64":
        audio, _ = audio_io.load_base64(item.media_type or "audio/wav", item.source)
    else:
        raise TypeError(f"unknown Audar-TTS reference source: {item.source_kind}")

    duration = len(audio) / REFERENCE_SAMPLE_RATE
    if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        raise ValueError(
            "Audar-TTS reference audio must be 5-15 seconds; "
            f"got {duration:.2f} seconds"
        )
    return torch.from_numpy(audio).float().reshape(1, 1, -1)


def _encode_reference(codec: Any, device: str, item: _ReferenceInput) -> torch.Tensor:
    waveform = _load_reference_waveform(item).to(device)
    with torch.inference_mode():
        codes = torch.as_tensor(codec.encode_code(waveform)).squeeze()
    if codes.ndim != 1 or codes.numel() == 0:
        raise RuntimeError(
            f"Audar-TTS codec returned invalid reference codes: {tuple(codes.shape)}"
        )
    return codes.detach().to(device="cpu", dtype=torch.long)


def create_preprocessing_executor() -> SimpleScheduler:
    return SimpleScheduler(
        lambda payload: _store_state(payload, build_audar_state(payload))
    )


def create_reference_encoder_executor(
    *,
    gpu_id: int | None = None,
    codec_model: str = DEFAULT_CODEC_MODEL,
    codec_revision: str = "main",
) -> SimpleScheduler:
    device = _device(gpu_id)
    codec = _load_codec(codec_model, codec_revision, device)

    def _encode(payload: StagePayload) -> StagePayload:
        state = _load_state(payload)
        codes = _encode_reference(
            codec, device, _normalize_reference(state.reference_audio)
        )
        state.prompt = build_prompt(
            state.target_text, state.reference_text, codes.tolist()
        )
        state.target_text = ""
        state.reference_text = ""
        state.reference_audio = None
        return _store_state(payload, state)

    return SimpleScheduler(_encode)


def _resolve_gguf(model_path: str, filename: str, revision: str) -> str:
    path = Path(model_path).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidate = path / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Audar-TTS GGUF not found: {candidate}")
        return str(candidate)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_path, filename=filename, revision=revision)


def create_tts_engine_executor(
    model_path: str,
    *,
    gpu_id: int | None = None,
    gguf_filename: str = DEFAULT_GGUF_FILENAME,
    model_revision: str = "main",
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
) -> SimpleScheduler:
    try:
        from llama_cpp import LLAMA_SPLIT_MODE_NONE, Llama
    except ImportError as exc:
        raise RuntimeError(
            "Audar-TTS requires the 'audar-tts' optional dependencies"
        ) from exc

    model_file = _resolve_gguf(model_path, gguf_filename, model_revision)
    llm = Llama(
        model_path=model_file,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        split_mode=LLAMA_SPLIT_MODE_NONE,
        main_gpu=int(gpu_id or 0),
        verbose=False,
    )
    stop_tokens = llm.tokenize(
        TARGET_CODES_END.encode("utf-8"), add_bos=False, special=True
    )
    if len(stop_tokens) != 1:
        raise RuntimeError(
            "Audar-TTS GGUF must encode <|TARGET_CODES_END|> as one token"
        )
    stop_token = stop_tokens[0]

    def _generate(payload: StagePayload) -> StagePayload:
        state = _load_state(payload)
        if not state.prompt:
            raise RuntimeError("Audar-TTS generation requires an encoded prompt")
        prompt_tokens = llm.tokenize(
            state.prompt.encode("utf-8"), add_bos=False, special=True
        )
        max_new_tokens = min(
            int(state.generation_kwargs["max_new_tokens"]),
            n_ctx - len(prompt_tokens),
        )
        if max_new_tokens <= 0:
            raise ValueError("Audar-TTS prompt exceeds the llama.cpp context window")
        llm.reset()
        seed = state.generation_kwargs.get("seed")
        if seed is not None:
            llm.set_seed(int(seed))

        generated: list[int] = []
        pieces: list[str] = []
        started = time.perf_counter()
        for token in llm.generate(
            prompt_tokens,
            temp=float(state.generation_kwargs["temperature"]),
            top_k=int(state.generation_kwargs["top_k"]),
            top_p=float(state.generation_kwargs["top_p"]),
            repeat_penalty=float(state.generation_kwargs["repetition_penalty"]),
        ):
            if token == stop_token or len(generated) >= max_new_tokens:
                break
            generated.append(int(token))
            pieces.append(
                llm.detokenize([token], special=True).decode("utf-8", "ignore")
            )
        state.engine_time_s = time.perf_counter() - started
        state.prompt_tokens = len(prompt_tokens)
        state.completion_tokens = len(generated)
        state.audio_codes = parse_speech_codes("".join(pieces))
        if not state.audio_codes:
            raise RuntimeError("Audar-TTS model emitted no speech tokens")
        state.prompt = None
        return _store_state(payload, state)

    return SimpleScheduler(_generate)


def create_vocoder_executor(
    *,
    gpu_id: int | None = None,
    codec_model: str = DEFAULT_CODEC_MODEL,
    codec_revision: str = "main",
) -> SimpleScheduler:
    device = _device(gpu_id)
    codec = _load_codec(codec_model, codec_revision, device)

    async def _decode(payload: StagePayload) -> StagePayload:
        state = _load_state(payload)
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if codes.ndim != 1 or codes.numel() == 0:
            raise RuntimeError("Audar-TTS vocoder requires non-empty audio codes")
        with torch.inference_mode():
            waveform = codec.decode_code(codes.to(device)[None, None, :])
        waveform = torch.as_tensor(waveform).detach().cpu().reshape(-1)
        state.audio_codes = None
        state.sample_rate = OUTPUT_SAMPLE_RATE
        _store_state(payload, state)
        payload.data.update(
            audio_waveform_payload(
                waveform,
                sample_rate=OUTPUT_SAMPLE_RATE,
                modality="audio",
                source_hint="Audar-TTS",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    return SimpleScheduler(_decode)


def _load_state(payload: StagePayload) -> AudarTTSState:
    return load_pipeline_state(payload, AudarTTSState)


def _store_state(payload: StagePayload, state: AudarTTSState) -> StagePayload:
    return store_pipeline_state(payload, state)
