# SPDX-License-Identifier: Apache-2.0
"""Per-request state for Audar-TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglang_omni.scheduling.pipeline_state import PipelineStateBase


@dataclass
class AudarTTSState(PipelineStateBase):
    target_text: str = ""
    reference_text: str = ""
    reference_audio: Any | None = None
    prompt: str | None = None
    audio_codes: Any | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 24000

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "generation_kwargs": dict(self.generation_kwargs),
            "sample_rate": int(self.sample_rate),
        }
        if self.target_text:
            data["target_text"] = self.target_text
        if self.reference_text:
            data["reference_text"] = self.reference_text
        if self.reference_audio is not None:
            data["reference_audio"] = self.reference_audio
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.audio_codes is not None:
            data["audio_codes"] = self.audio_codes
        self.append_usage_fields(data)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "AudarTTSState":
        if not isinstance(data, dict):
            data = {}
        generation_kwargs = data.get("generation_kwargs")
        return cls(
            target_text=str(data.get("target_text", "")),
            reference_text=str(data.get("reference_text", "")),
            reference_audio=data.get("reference_audio"),
            prompt=data.get("prompt"),
            audio_codes=data.get("audio_codes"),
            generation_kwargs=(
                dict(generation_kwargs) if isinstance(generation_kwargs, dict) else {}
            ),
            sample_rate=int(data.get("sample_rate", 24000) or 24000),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
        )
