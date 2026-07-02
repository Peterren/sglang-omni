# SPDX-License-Identifier: Apache-2.0
"""Capability declarations for model architectures."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class ModelCapabilities:
    """Static capability declaration for a TTS model architecture.

    These flags describe what a model architecture can support, not what a
    specific pipeline instance enables for a checkpoint or deployment.
    """

    supports_reference_audio: bool
    supports_batch_vocoder: bool
    supports_streaming_vocoder: bool
    supports_cuda_graph: bool
    supports_torch_compile: bool


def get_model_capabilities(architecture: str) -> ModelCapabilities | None:
    """Look up capabilities for a registered model architecture."""
    module = _model_package_for_architecture(architecture)
    if module is None:
        return None
    return _module_model_capabilities(module)


def _model_package_for_architecture(architecture: str) -> ModuleType | None:
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    config_cls = PIPELINE_CONFIG_REGISTRY.configs.get(architecture)
    if config_cls is None:
        return None
    package = config_cls.__module__.rsplit(".", 1)[0]
    return importlib.import_module(package)


def _module_model_capabilities(module: ModuleType) -> ModelCapabilities | None:
    capabilities = getattr(module, "CAPABILITIES", None)
    if capabilities is None:
        return None
    return _ensure_model_capabilities(
        capabilities, f"{module.__name__}.CAPABILITIES"
    )


def _ensure_model_capabilities(
    capabilities: object, source: str
) -> ModelCapabilities:
    if not isinstance(capabilities, ModelCapabilities):
        raise TypeError(f"{source} must be a ModelCapabilities instance")
    return capabilities


__all__ = [
    "ModelCapabilities",
    "get_model_capabilities",
]
