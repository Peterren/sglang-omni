# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ARK-ASR-3B adapter (CPU-only, no checkpoint download)."""

import inspect

import torch
from transformers import WhisperConfig

from sglang_omni.models.arkasr.audio_lengths import (
    arkasr_audio_token_lengths,
    arkasr_num_audio_tokens,
)
from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter, ArkAudioTower
from sglang_omni.models.arkasr.config import ArkasrPipelineConfig
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
from sglang_omni.models.arkasr.request_builders import _build_suppressed_token_ids
from sglang_omni.models.arkasr.stages import create_sglang_arkasr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def _tiny_config():
    """Small ARK config for CPU shape tests (no checkpoint)."""
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=64,
    )
    return ArkasrConfig(
        whisper_config=whisper,
        merge_factor=4,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=256,
        audio_token_id=151663,
    )


def test_arkasr_config_registered():
    config = ArkasrPipelineConfig(model_path="AutoArk-AI/ARK-ASR-3B")
    assert config.entry_stage == "asr"
    assert config.stages[0].name == "asr"
    assert config.stages[0].terminal
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("ArkasrForConditionalGeneration")
        is ArkasrPipelineConfig
    )


def test_arkasr_stage_defaults():
    signature = inspect.signature(create_sglang_arkasr_executor)
    assert signature.parameters["max_running_requests"].default == 32
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16


def test_arkasr_audio_token_count():
    # matches checkpoint processor: (mel+1)//2 // merge_factor(4)
    assert arkasr_num_audio_tokens(400) == 50  # (401//2)=200 -> 50
    assert arkasr_num_audio_tokens(3000) == 375  # (3001//2)=1500 -> 375
    assert arkasr_num_audio_tokens(1) == 1  # floor at 1
    # list form
    assert arkasr_audio_token_lengths([400, 8]) == [50, 1]


def test_arkasr_config_text_config_is_self():
    cfg = _tiny_config()
    # ARK's LM params live at top level; text_config must resolve to the config
    # itself so sglang-omni's _ARCH_CONFIG_MAP reads the right dims.
    assert cfg.text_config is cfg
    assert cfg.get_text_config() is cfg
    assert cfg.text_config.num_attention_heads == 4
    assert cfg.text_config.hidden_size == 48


def test_ark_audio_tower_forward_shape():
    torch.manual_seed(0)
    cfg = _tiny_config()
    adapter = ArkAudioMLPAdapter(cfg).eval()
    mel_frames = 40  # -> conv2 stride2 -> ~20 -> merge4 -> ~5 audio tokens
    mel = torch.randn(1, cfg.whisper_config.num_mel_bins, mel_frames)
    with torch.no_grad():
        out = adapter(mel)
    assert out.dim() == 3
    assert out.size(0) == 1
    assert out.size(-1) == cfg.hidden_size  # projected to LLM hidden
    assert out.size(1) >= 1


def test_ark_tower_rope_toggle():
    cfg = _tiny_config()
    tower = ArkAudioTower(cfg)
    assert tower.use_rope is True
    assert hasattr(tower, "rotary_embedding")


def test_ark_suppressed_token_ids():
    """The generation-suppression set must cover every ``<...>`` added/special
    token except EOS (mirrors the checkpoint's bad_words_ids), so markers like
    <|audio|> / <tool_call> cannot leak into transcripts."""

    class _FakeTok:
        eos_token_id = 100
        all_special_ids = [100, 101, 102]

        def get_added_vocab(self):
            return {
                "<|im_start|>": 101,
                "<|audio|>": 103,
                "<tool_call>": 104,
                "hello": 105,  # normal token: must NOT be suppressed
                "</tool_call>": 106,
            }

    ids = _build_suppressed_token_ids(_FakeTok())
    assert 100 not in ids  # EOS kept
    assert 101 in ids and 102 in ids  # special ids
    assert 103 in ids and 104 in ids and 106 in ids  # <...> added tokens
    assert 105 not in ids  # normal token untouched
    assert ids == sorted(ids)  # deterministic order
