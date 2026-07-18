# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io
import sys
import types
import wave
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.models.audar_tts import stages
from sglang_omni.models.audar_tts.config import AudarTTSPipelineConfig
from sglang_omni.models.audar_tts.payload_types import AudarTTSState
from sglang_omni.models.audar_tts.protocol import build_prompt, parse_speech_codes
from sglang_omni.models.audar_tts.request_builders import build_audar_state
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.protocol import SUPPORTED_TTS_LANGUAGES


class FakeCodec:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0
        self.device = "cpu"

    def eval(self) -> "FakeCodec":
        return self

    def to(self, device: str) -> "FakeCodec":
        self.device = device
        return self

    def encode_code(self, waveform: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        assert waveform.shape == (1, 1, 80000)
        return torch.tensor([[[7, 8, 9]]])

    def decode_code(self, codes: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        assert codes.ndim == 3
        return torch.tensor([[[0.25, -0.5, 0.75]]])


def make_payload(
    *,
    inputs: Any = "",
    params: dict[str, Any] | None = None,
    tts_params: dict[str, Any] | None = None,
    state: AudarTTSState | None = None,
    request_id: str = "request",
) -> StagePayload:
    metadata = {"tts_params": tts_params} if tts_params is not None else {}
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=inputs, params=params or {}, metadata=metadata),
        data=state.to_dict() if state is not None else {},
    )


def five_second_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(80000, dtype="<i2").tobytes())
    return output.getvalue()


def test_prompt_matches_official_audar_protocol() -> None:
    prompt = build_prompt("مرحبا", "صوت مرجعي", [7, 8])

    assert prompt == (
        "user: Convert the text to speech:"
        "<|REF_TEXT_START|>صوت مرجعي<|REF_TEXT_END|>"
        "<|REF_SPEECH_START|><|speech_7|><|speech_8|><|REF_SPEECH_END|>"
        "<|TARGET_TEXT_START|>مرحبا<|TARGET_TEXT_END|>"
        "\nassistant:<|TARGET_CODES_START|>"
    )
    assert parse_speech_codes("x<|speech_5|><|speech_42|>y") == [5, 42]


def test_config_and_state_contracts() -> None:
    config = AudarTTSPipelineConfig(model_path="audarai/Audar-TTS-V1-Turbo")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "reference_encoder",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.supports_uploaded_voice_references() is True
    assert "Arabic" in SUPPORTED_TTS_LANGUAGES
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("AudarTTSForConditionalGeneration")
        is AudarTTSPipelineConfig
    )

    state = AudarTTSState(
        target_text="target",
        reference_text="reference",
        reference_audio={"bytes": b"wav"},
        prompt="prompt",
        audio_codes=[1, 2],
        generation_kwargs={"temperature": 0.7},
        prompt_tokens=10,
        completion_tokens=20,
        engine_time_s=0.5,
    )
    assert AudarTTSState.from_dict(state.to_dict()) == state


def test_request_lowering_keeps_audar_defaults_unless_explicit() -> None:
    reference = {"bytes": five_second_wav(), "text": "reference transcript"}
    implicit = make_payload(
        inputs={"text": "target", "references": [reference]},
        params={
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
        },
    )
    implicit_state = build_audar_state(implicit)
    assert implicit_state.generation_kwargs == {
        "max_new_tokens": 2048,
        "temperature": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }

    explicit = make_payload(
        inputs={"text": "target", "references": [reference]},
        params={"temperature": 0.6, "top_k": 20, "max_new_tokens": 128},
        tts_params={
            "explicit_generation_params": [
                "temperature",
                "top_k",
                "max_new_tokens",
            ],
            "seed": 17,
        },
    )
    explicit_state = build_audar_state(explicit)
    assert explicit_state.generation_kwargs == {
        "max_new_tokens": 128,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "seed": 17,
    }


def test_request_lowering_requires_one_transcribed_reference() -> None:
    with pytest.raises(ValueError, match="reference audio"):
        build_audar_state(make_payload(inputs="target"))
    with pytest.raises(ValueError, match="reference transcript"):
        build_audar_state(
            make_payload(
                inputs={
                    "text": "target",
                    "references": [{"bytes": five_second_wav()}],
                }
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_audar_state(
            make_payload(
                inputs={
                    "text": "target",
                    "references": [
                        {"bytes": b"a", "text": "a"},
                        {"bytes": b"b", "text": "b"},
                    ],
                }
            )
        )


def test_reference_encoder_builds_prompt_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    monkeypatch.setattr(stages, "_load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_reference_encoder_executor(gpu_id=None)
    reference_audio = {"bytes": five_second_wav()}

    def encode(request_id: str) -> AudarTTSState:
        payload = make_payload(
            state=AudarTTSState(
                target_text="target",
                reference_text="reference",
                reference_audio=reference_audio,
            ),
            request_id=request_id,
        )
        return AudarTTSState.from_dict(scheduler._fn(payload).data)

    first = encode("first")
    second = encode("second")

    assert scheduler._max_concurrency == 1
    assert codec.encode_calls == 2
    assert first.prompt == build_prompt("target", "reference", [7, 8, 9])
    assert second.prompt == first.prompt
    assert first.reference_audio is None


def test_codec_model_is_shared_between_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    loads = 0

    class FakeNeuCodec:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeCodec:
            nonlocal loads
            loads += 1
            return codec

    monkeypatch.setitem(
        sys.modules, "neucodec", types.SimpleNamespace(NeuCodec=FakeNeuCodec)
    )
    stages._load_codec.cache_clear()
    try:
        first = stages._load_codec("codec", "revision", "cpu")
        second = stages._load_codec("codec", "revision", "cpu")
    finally:
        stages._load_codec.cache_clear()

    assert first is second is codec
    assert loads == 1


def test_llama_cpp_stage_matches_official_generation_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLlama:
        instance: "FakeLlama"

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.generate_kwargs: dict[str, Any] | None = None
            self.seed: int | None = None
            self.reset_calls = 0
            FakeLlama.instance = self

        def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
            assert add_bos is False
            assert special is True
            return [99] if text == b"<|TARGET_CODES_END|>" else [1, 2, 3]

        def generate(self, tokens: list[int], **kwargs: Any):
            assert tokens == [1, 2, 3]
            self.generate_kwargs = kwargs
            yield 10
            yield 11
            yield 99

        def detokenize(self, tokens: list[int], *, special: bool) -> bytes:
            assert special is True
            return {10: b"<|speech_123|>", 11: b"<|speech_456|>"}[tokens[0]]

        def set_seed(self, seed: int) -> None:
            self.seed = seed

        def reset(self) -> None:
            self.reset_calls += 1

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        types.SimpleNamespace(LLAMA_SPLIT_MODE_NONE=0, Llama=FakeLlama),
    )
    monkeypatch.setattr(stages, "_resolve_gguf", lambda *args: "/model.gguf")
    payload = make_payload(
        state=AudarTTSState(
            prompt="prompt",
            generation_kwargs={
                "max_new_tokens": 16,
                "temperature": 1.0,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "seed": 23,
            },
        )
    )

    scheduler = stages.create_tts_engine_executor(
        "audarai/Audar-TTS-V1-Turbo", gpu_id=2
    )
    result = AudarTTSState.from_dict(scheduler._fn(payload).data)

    assert result.audio_codes == [123, 456]
    assert result.prompt is None
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert FakeLlama.instance.seed == 23
    assert FakeLlama.instance.reset_calls == 1
    assert FakeLlama.instance.kwargs["main_gpu"] == 2
    assert FakeLlama.instance.generate_kwargs == {
        "temp": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }


def test_vocoder_emits_24khz_audio_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = FakeCodec()
    monkeypatch.setattr(stages, "_load_codec", lambda *args, **kwargs: codec)
    scheduler = stages.create_vocoder_executor(gpu_id=None)
    payload = make_payload(
        state=AudarTTSState(
            audio_codes=[1, 2],
            prompt_tokens=3,
            completion_tokens=2,
            engine_time_s=0.25,
        )
    )

    result = asyncio.run(scheduler._fn(payload))

    assert scheduler._batch_fn is None
    assert codec.decode_calls == 1
    assert result.data["audio_waveform_shape"] == [3]
    assert result.data["audio_waveform_dtype"] == "float32"
    assert result.data["sample_rate"] == 24000
    assert result.data["modality"] == "audio"
    assert "audio_codes" not in result.data
    assert result.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "engine_time_s": 0.25,
    }
