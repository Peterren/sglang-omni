# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Voxtral TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.tts_streaming import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline.state_io import load_state, store_state
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


def _ensure_non_empty_audio_codes(audio_codes: Any) -> None:
    if audio_codes is None:
        raise ValueError("Voxtral TTS generated no audio codes")
    if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() == 0:
        raise ValueError("Voxtral TTS generated no audio codes")
    if isinstance(audio_codes, (list, tuple)) and len(audio_codes) == 0:
        raise ValueError("Voxtral TTS generated no audio codes")


def _as_code_tensor(
    audio_codes: Any,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _ensure_non_empty_audio_codes(audio_codes)
    if isinstance(audio_codes, torch.Tensor):
        tensor = audio_codes.detach()
    else:
        tensor = torch.as_tensor(audio_codes)
    if tensor.numel() == 0:
        raise ValueError("Voxtral TTS generated no audio codes")
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        raise ValueError(
            f"Voxtral audio codes must be 2-D [frames, codebooks], "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.to(device=device, dtype=torch.long)


def _to_audio_tensor(audio: Any) -> torch.Tensor:
    if isinstance(audio, torch.Tensor):
        return audio.detach().float().reshape(-1)
    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1)


def _apply_fade_in(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    fade_in_ms: int,
) -> torch.Tensor:
    fade_samples = min(int(fade_in_ms * sample_rate / 1000), int(audio.numel()))
    if fade_samples <= 0:
        return audio
    fade_in = torch.linspace(
        0,
        1,
        fade_samples,
        device=audio.device,
        dtype=audio.dtype,
    )
    audio = audio.clone()
    audio[:fade_samples] = audio[:fade_samples] * fade_in
    return audio


def _decode_voxtral_codes(
    audio_tokenizer: Any,
    audio_codes: torch.Tensor,
    *,
    include_start_warmup: bool,
    fade_in_ms: int = 0,
) -> torch.Tensor:
    n_warmup = 2
    warmup_samples = 0
    codes = audio_codes
    if include_start_warmup and codes.shape[0] > 0:
        warmup = codes[0:1].repeat(n_warmup, 1)
        codes = torch.cat([warmup, codes], dim=0)
        warmup_samples = int(n_warmup * audio_tokenizer.downsample_factor)

    results = audio_tokenizer.decode_helper_batch_async([codes])
    audio = _to_audio_tensor(results[0])

    if warmup_samples > 0 and int(audio.numel()) > warmup_samples:
        audio = audio[warmup_samples:]
    if fade_in_ms > 0:
        audio = _apply_fade_in(
            audio,
            sample_rate=int(audio_tokenizer.sampling_rate),
            fade_in_ms=fade_in_ms,
        )
    return audio.contiguous()


def build_voxtral_audio_payload(
    audio_tokenizer: Any,
    audio_codes: Any,
    *,
    device: torch.device | str | None = None,
    fade_in_ms: int = 10,
    source_hint: str = "Voxtral TTS",
) -> dict[str, Any]:
    codes = _as_code_tensor(audio_codes, device=device)
    audio = _decode_voxtral_codes(
        audio_tokenizer,
        codes,
        include_start_warmup=True,
        fade_in_ms=fade_in_ms,
    )
    return audio_waveform_payload(
        audio,
        sample_rate=int(audio_tokenizer.sampling_rate),
        modality="audio",
        source_hint=source_hint,
    )


def store_voxtral_audio_result(
    payload: StagePayload,
    state: VoxtralTTSState,
    audio_payload: dict[str, Any],
    *,
    sample_rate: int,
) -> StagePayload:
    state.audio_samples = None
    state.sample_rate = int(sample_rate)
    payload = store_state(payload, state)

    payload.data.update(audio_payload)
    payload.data["sample_rate"] = int(sample_rate)
    payload.data["modality"] = "audio"

    usage = build_voxtral_usage(state)
    if usage is not None:
        payload.data["usage"] = usage
    return payload


def build_voxtral_usage(state: VoxtralTTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens):
        return None
    return {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }


@dataclass
class _VoxtralStreamState:
    codes: list[torch.Tensor] = field(default_factory=list)
    emitted_frames: int = 0
    next_decode_frames: int = 0
    has_emitted: bool = False
    initial_codec_chunk_frames: int = 0
    num_codebooks: int | None = None


class VoxtralStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode Voxtral codec frames incrementally for streaming speech."""

    def __init__(
        self,
        audio_tokenizer: Any,
        *,
        device: torch.device | str = "cuda:0",
        stream_stride: int = 10,
        stream_followup_stride: int = 30,
        stream_overlap_frames: int = 2,
        fade_in_ms: int = 10,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_frames < 0:
            raise ValueError("stream_overlap_frames must be >= 0")
        if fade_in_ms < 0:
            raise ValueError("fade_in_ms must be >= 0")

        self._audio_tokenizer = audio_tokenizer
        self._device = torch.device(device)
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_frames = int(stream_overlap_frames)
        self._fade_in_ms = int(fade_in_ms)
        self._sample_rate = int(audio_tokenizer.sampling_rate)
        self._stream_states: dict[str, _VoxtralStreamState] = {}

        super().__init__(self._vocode_payload)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"Voxtral request params must be a dict, got "
                f"{type(params).__name__}"
            )
        return bool(params.get("stream", False))

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        state = load_state(payload)
        _ensure_non_empty_audio_codes(state.audio_codes)

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = self._stream_states.setdefault(request_id, _VoxtralStreamState())
        self._latch_initial_codec_chunk_frames_from_mapping(
            request_id,
            state,
            (
                payload.request.params
                if isinstance(payload.request.params, dict)
                else None
            ),
        )

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self._stream_states.setdefault(request_id, _VoxtralStreamState())
        self._latch_stream_metadata(request_id, state, item.metadata)

        row = self._stream_row_tensor(request_id, item.data)
        eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        if int(row[0].item()) == eos_id:
            return []
        if state.num_codebooks is not None and int(row.shape[0]) != state.num_codebooks:
            raise ValueError(
                f"Voxtral stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {state.num_codebooks}"
            )

        state.codes.append(row)
        output = self._decode_delta(state, is_final=False)
        if output is None:
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=output,
                metadata={"modality": "audio"},
            )
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads[request_id]
        state = self._stream_states.setdefault(request_id, _VoxtralStreamState())

        output = self._decode_delta(state, is_final=True)
        if output is None and not state.has_emitted:
            output = self._audio_payload_from_stage_payload(payload)

        messages: list[OutgoingMessage] = []
        if output is not None:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_voxtral_usage(load_state(payload))
        if usage is not None:
            final_data["usage"] = usage
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)

    def _latch_stream_metadata(
        self,
        request_id: str,
        state: _VoxtralStreamState,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not isinstance(metadata, dict):
            return
        if metadata.get("modality") not in (None, "audio_codes"):
            raise ValueError(
                f"Voxtral stream chunk modality must be audio_codes, got "
                f"{metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"Voxtral stream chunk for {request_id!r} must include "
                "metadata['stream'] == True"
            )
        if "num_codebooks" in metadata:
            self._latch_num_codebooks(
                request_id,
                state,
                metadata["num_codebooks"],
            )
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            self._latch_initial_codec_chunk_frames_from_mapping(
                request_id,
                state,
                metadata,
            )

    @staticmethod
    def _latch_num_codebooks(
        request_id: str,
        state: _VoxtralStreamState,
        num_codebooks: Any,
    ) -> None:
        try:
            value = int(num_codebooks)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Voxtral stream chunk for {request_id!r} must include integer "
                "num_codebooks"
            ) from exc
        if value <= 0:
            raise ValueError(
                f"Voxtral stream chunk for {request_id!r} has invalid "
                f"num_codebooks={value}"
            )
        if state.num_codebooks is not None and state.num_codebooks != value:
            raise ValueError(
                f"Voxtral stream num_codebooks changed for {request_id!r}: "
                f"{state.num_codebooks} -> {value}"
            )
        state.num_codebooks = value

    def _latch_initial_codec_chunk_frames_from_mapping(
        self,
        request_id: str,
        state: _VoxtralStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        del request_id
        state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
            params,
            steady_chunk_frames=self._stream_stride,
        )

    def _stream_row_tensor(self, request_id: str, row: Any) -> torch.Tensor:
        if isinstance(row, torch.Tensor):
            tensor = row.detach()
        else:
            tensor = torch.as_tensor(row)
        tensor = tensor.to(device=self._device, dtype=torch.long, non_blocking=True)
        if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tensor.ndim != 1:
            raise ValueError(
                f"Voxtral stream chunk for {request_id!r} must be 1-D "
                f"[codebooks], got {tuple(tensor.shape)}"
            )
        if tensor.numel() == 0:
            raise ValueError(f"Voxtral stream chunk for {request_id!r} is empty")
        return tensor

    def _decode_delta(
        self,
        state: _VoxtralStreamState,
        *,
        is_final: bool,
    ) -> dict[str, Any] | None:
        total_frames = len(state.codes)
        if total_frames == 0:
            return None

        use_initial_chunk = (
            state.initial_codec_chunk_frames > 0
            and state.initial_codec_chunk_frames < self._stream_stride
            and not state.has_emitted
        )
        next_decode_frames = state.next_decode_frames or (
            state.initial_codec_chunk_frames
            if use_initial_chunk and not is_final
            else self._stream_stride
        )
        if not is_final and total_frames < next_decode_frames:
            state.next_decode_frames = next_decode_frames
            return None

        emit_until = total_frames
        if use_initial_chunk and not is_final:
            emit_until = min(total_frames, state.initial_codec_chunk_frames)
        if emit_until <= state.emitted_frames:
            state.next_decode_frames = total_frames + self._stream_followup_stride
            return None

        window_start = max(0, state.emitted_frames - self._stream_overlap_frames)
        window = torch.stack(state.codes[window_start:emit_until], dim=0).to(
            device=self._device,
            dtype=torch.long,
        )
        audio = _decode_voxtral_codes(
            self._audio_tokenizer,
            window,
            include_start_warmup=window_start == 0,
            fade_in_ms=0,
        )

        trim_frames = state.emitted_frames - window_start
        trim_samples = int(trim_frames * self._audio_tokenizer.downsample_factor)
        if trim_samples > 0:
            audio = audio[min(trim_samples, int(audio.numel())) :]
        if audio.numel() == 0:
            state.next_decode_frames = total_frames + self._stream_followup_stride
            return None

        if not state.has_emitted and self._fade_in_ms > 0:
            audio = _apply_fade_in(
                audio,
                sample_rate=self._sample_rate,
                fade_in_ms=self._fade_in_ms,
            )

        state.emitted_frames = emit_until
        state.next_decode_frames = emit_until + self._stream_followup_stride
        state.has_emitted = True
        return audio_waveform_payload(
            audio,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Voxtral TTS streaming",
        )

    def _audio_payload_from_stage_payload(
        self,
        payload: StagePayload,
    ) -> dict[str, Any]:
        state = load_state(payload)
        return build_voxtral_audio_payload(
            self._audio_tokenizer,
            state.audio_codes,
            device=self._device,
            fade_in_ms=self._fade_in_ms,
            source_hint="Voxtral TTS streaming",
        )

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        audio_payload = build_voxtral_audio_payload(
            self._audio_tokenizer,
            state.audio_codes,
            device=self._device,
            fade_in_ms=self._fade_in_ms,
        )
        return store_voxtral_audio_result(
            payload,
            state,
            audio_payload,
            sample_rate=self._sample_rate,
        )


__all__ = [
    "VoxtralStreamingVocoderScheduler",
    "build_voxtral_audio_payload",
    "store_voxtral_audio_result",
    "_ensure_non_empty_audio_codes",
]
