# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize model support."""

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=True,
    supports_sglang_piecewise_prefill=True,
    supports_torch_compile=False,
)

__all__ = ["CAPABILITIES"]
