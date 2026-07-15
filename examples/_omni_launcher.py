# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import sys
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _add_model_path(
    parser: argparse.ArgumentParser,
    default: str,
) -> None:
    parser.add_argument(
        "--model-path",
        type=str,
        default=default,
        help="Hugging Face model id or local path",
    )


def _add_relay_backend(
    parser: argparse.ArgumentParser,
    *,
    choices: Sequence[str] = ("nixl", "shm"),
) -> None:
    parser.add_argument(
        "--relay-backend",
        type=str,
        default="shm",
        choices=choices,
        help="Relay backend for inter-stage data transfer",
    )


def _add_mem_fraction(
    parser: argparse.ArgumentParser,
    help_text: str,
) -> None:
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=help_text,
    )


def _add_server_args(
    parser: argparse.ArgumentParser,
    *,
    model_name: str | None,
) -> None:
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default=model_name)


def _validate_fraction(flag_name: str, value: float | None) -> None:
    if value is not None and not 0.0 < value < 1.0:
        raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")


def _apply_stage_factory_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object] | None = None,
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue
        factory_args = dict(stage.factory_args or {})
        if updates:
            factory_args.update(updates)
        if server_arg_updates:
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(server_arg_updates)
            factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args
        return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _set_stage_gpu(
    config: Any,
    stage_name: str,
    gpu_id: int | list[int],
) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = gpu_id
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _set_stage_tp_size(config: Any, stage_name: str, tp_size: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.tp_size = int(tp_size)
            stage.parallelism = stage.parallelism.model_copy(
                update={"tp": int(tp_size)}
            )
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _parse_thinker_tp_gpu_list(spec: str, tp_size: int) -> list[int]:
    try:
        gpu_ids = [int(piece.strip()) for piece in spec.split(",") if piece.strip()]
    except ValueError as exc:
        raise ValueError(
            f"--gpu-thinker-tp must be a comma-separated list of integers, got {spec!r}"
        ) from exc
    if any(gpu < 0 for gpu in gpu_ids):
        raise ValueError(f"--gpu-thinker-tp GPU ids must be >= 0, got {gpu_ids}")
    if len(gpu_ids) != tp_size:
        raise ValueError(
            f"--gpu-thinker-tp has {len(gpu_ids)} entries but --thinker-tp-size="
            f"{tp_size} requires exactly {tp_size}"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-thinker-tp must list distinct GPU ids, got {gpu_ids}")
    return gpu_ids


def _resolve_speech_mem_fractions(
    config: Any,
    *,
    global_mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
    talker_mem_fraction_static: float | None,
) -> None:
    values = (
        ("--mem-fraction-static", global_mem_fraction_static),
        ("--thinker-mem-fraction-static", thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", talker_mem_fraction_static),
    )
    for flag_name, value in values:
        _validate_fraction(flag_name, value)
    thinker_value = (
        thinker_mem_fraction_static
        if thinker_mem_fraction_static is not None
        else global_mem_fraction_static
    )
    talker_value = (
        talker_mem_fraction_static
        if talker_mem_fraction_static is not None
        else global_mem_fraction_static
    )
    if thinker_value is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"mem_fraction_static": thinker_value},
        )
    if talker_value is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"mem_fraction_static": talker_value},
        )


def _build_qwen_text_server_parser() -> argparse.ArgumentParser:
    parser = _parser("Launch Qwen3-Omni with text-only OpenAI responses.")
    _add_model_path(parser, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--thinker-max-seq-len", type=int, default=None)
    parser.add_argument("--cpu-offload-gb", type=int, default=0)
    _add_relay_backend(parser, choices=("shm", "nccl", "nixl"))
    _add_mem_fraction(
        parser,
        "Set mem_fraction_static for the thinker stage.",
    )
    parser.add_argument(
        "--encoder-mem-reserve",
        type=float,
        default=None,
        help="GPU-memory fraction reserved for colocated encoders.",
    )
    _add_server_args(parser, model_name=None)
    parser.add_argument("--enable-realtime", action="store_true")
    return parser


def launch_qwen_text_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
    from sglang_omni.serve import launch_server

    if args.mem_fraction_static is not None and args.encoder_mem_reserve is not None:
        raise ValueError(
            "--mem-fraction-static and --encoder-mem-reserve are mutually exclusive"
        )
    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    if (
        args.encoder_mem_reserve is not None
        and not 0.0 <= args.encoder_mem_reserve < 1.0
    ):
        raise ValueError(
            f"--encoder-mem-reserve must be in [0, 1), got {args.encoder_mem_reserve}"
        )

    config = Qwen3OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    thinker_updates: dict[str, object] = {}
    preprocessing_updates: dict[str, object] = {}
    if args.thinker_max_seq_len is not None:
        max_seq_len = int(args.thinker_max_seq_len)
        thinker_updates["thinker_max_seq_len"] = max_seq_len
        preprocessing_updates["thinker_max_seq_len"] = max_seq_len
    if args.encoder_mem_reserve is not None:
        thinker_updates["encoder_mem_reserve"] = args.encoder_mem_reserve

    server_updates: dict[str, object] = {}
    if args.cpu_offload_gb:
        server_updates["cpu_offload_gb"] = int(args.cpu_offload_gb)
    if args.mem_fraction_static is not None:
        server_updates["mem_fraction_static"] = args.mem_fraction_static
    if thinker_updates or server_updates:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            updates=thinker_updates,
            server_arg_updates=server_updates,
        )
    if preprocessing_updates:
        _apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=preprocessing_updates,
        )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        enable_realtime=args.enable_realtime,
    )


def _build_ming_text_server_parser() -> argparse.ArgumentParser:
    parser = _parser("Launch Ming-Omni with text-only OpenAI responses.")
    _add_model_path(parser, "inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-audio-encoder", type=int, default=None)
    parser.add_argument("--gpu-image-encoder", type=int, nargs="+", default=None)
    parser.add_argument("--image-encoder-tp", type=int, default=1)
    parser.add_argument("--thinker-only", action="store_true")
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--cpu-offload-gb", type=int, default=80)
    _add_mem_fraction(parser, "Set mem_fraction_static for the thinker stage.")
    _add_relay_backend(parser, choices=("shm", "nccl", "nixl"))
    _add_server_args(parser, model_name="ming-omni")
    return parser


def _configure_ming_thinker_only(config: Any) -> None:
    stages = {stage.name: stage for stage in config.stages}
    preprocessing = stages["preprocessing"]
    aggregate = stages["mm_aggregate"]
    preprocessing.next = "mm_aggregate"
    preprocessing.project_payload = {
        "mm_aggregate": (
            "sglang_omni.models.ming_omni.stages.project_preprocessing_to_mm_aggregate"
        )
    }
    aggregate.wait_for = ["preprocessing"]
    config.stages = [
        stage
        for stage in config.stages
        if stage.name not in {"audio_encoder", "image_encoder"}
    ]


def launch_ming_text_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
    from sglang_omni.serve import launch_server

    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    config = MingOmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    if args.thinker_only:
        if args.gpu_audio_encoder is not None or args.gpu_image_encoder is not None:
            raise ValueError(
                "--gpu-audio-encoder/--gpu-image-encoder cannot be used "
                "with --thinker-only"
            )
        _configure_ming_thinker_only(config)

    server_updates: dict[str, object] = {}
    if args.tp_size and args.tp_size > 1:
        _set_stage_tp_size(config, "thinker", args.tp_size)
        _set_stage_gpu(config, "thinker", list(range(int(args.tp_size))))
        server_updates["disable_custom_all_reduce"] = True
    if args.gpu_audio_encoder is not None:
        _set_stage_gpu(config, "audio_encoder", int(args.gpu_audio_encoder))

    image_tp = int(args.image_encoder_tp)
    if image_tp < 1:
        raise ValueError("--image-encoder-tp must be >= 1")
    if image_tp > 1 and args.thinker_only:
        raise ValueError("--thinker-only cannot be used with --image-encoder-tp > 1")
    if image_tp > 1:
        if args.gpu_image_encoder is None:
            raise ValueError(
                "--gpu-image-encoder must be specified when --image-encoder-tp > 1"
            )
        if len(args.gpu_image_encoder) != image_tp:
            raise ValueError(
                f"--gpu-image-encoder requires exactly {image_tp} GPU ids "
                f"(matching --image-encoder-tp), got {len(args.gpu_image_encoder)}"
            )
        if len(set(args.gpu_image_encoder)) != len(args.gpu_image_encoder):
            raise ValueError("--gpu-image-encoder GPU ids must be unique")
        _set_stage_tp_size(config, "image_encoder", image_tp)
        _set_stage_gpu(config, "image_encoder", args.gpu_image_encoder)
    elif args.gpu_image_encoder is not None:
        _set_stage_gpu(config, "image_encoder", int(args.gpu_image_encoder[0]))

    if args.quantization:
        server_updates["quantization"] = args.quantization
    if args.cpu_offload_gb:
        server_updates["cpu_offload_gb"] = int(args.cpu_offload_gb)
    if args.mem_fraction_static is not None:
        server_updates["mem_fraction_static"] = args.mem_fraction_static
    _apply_stage_factory_updates(
        config,
        stage_name="thinker",
        updates={"thinker_max_seq_len": int(args.thinker_max_seq_len)},
        server_arg_updates=server_updates,
    )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def _add_qwen_speech_mem_args(parser: argparse.ArgumentParser) -> None:
    _add_mem_fraction(parser, "Set mem_fraction_static for both Qwen AR stages.")
    parser.add_argument("--thinker-mem-fraction-static", type=float, default=None)
    parser.add_argument("--talker-mem-fraction-static", type=float, default=None)


def _build_qwen_speech_server_parser() -> argparse.ArgumentParser:
    parser = _parser("Launch Qwen3-Omni with text and audio OpenAI responses.")
    _add_model_path(parser, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=None)
    parser.add_argument("--gpu-code-predictor", type=int, default=None)
    parser.add_argument("--gpu-code2wav", type=int, default=None)
    parser.add_argument("--gpu-image-encoder", type=int, default=None)
    parser.add_argument("--gpu-audio-encoder", type=int, default=None)
    parser.add_argument("--thinker-tp-size", type=int, default=1)
    parser.add_argument("--gpu-thinker-tp", type=str, default=None)
    _add_relay_backend(parser)
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--talker-max-seq-len", type=int, default=None)
    _add_qwen_speech_mem_args(parser)
    parser.add_argument(
        "--enable-partial-start",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--partial-start-min-chunks", type=int, default=5)
    parser.add_argument("--colocated", action="store_true")
    _add_server_args(parser, model_name="qwen3-omni")
    return parser


def launch_qwen_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import (
        MIN_PARTIAL_START_CHUNKS,
        Qwen3OmniSpeechColocatedPipelineConfig,
        Qwen3OmniSpeechPipelineConfig,
    )
    from sglang_omni.serve import launch_server
    from sglang_omni.utils.gpu_compat import should_disable_custom_all_reduce_for_gpus

    for flag_name, value in (
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--thinker-mem-fraction-static", args.thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", args.talker_mem_fraction_static),
    ):
        _validate_fraction(flag_name, value)

    enable_partial_start = (
        not args.colocated
        if args.enable_partial_start is None
        else bool(args.enable_partial_start)
    )
    if (
        enable_partial_start
        and args.partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS
    ):
        raise ValueError(
            f"--partial-start-min-chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
            f"got {args.partial_start_min_chunks}"
        )

    gpu_talker = args.gpu_talker
    if gpu_talker is None:
        gpu_talker = args.gpu_thinker if args.colocated else 1
    gpu_code2wav = args.gpu_code2wav
    if gpu_code2wav is None:
        gpu_code2wav = args.gpu_thinker if args.colocated else 0
    gpu_image_encoder = args.gpu_image_encoder
    if gpu_image_encoder is None:
        gpu_image_encoder = args.gpu_thinker if args.colocated else 0
    gpu_audio_encoder = args.gpu_audio_encoder
    if gpu_audio_encoder is None:
        gpu_audio_encoder = args.gpu_thinker if args.colocated else 0

    if args.colocated:
        colocated_gpus = {
            "--gpu-thinker": args.gpu_thinker,
            "--gpu-talker": gpu_talker,
            "--gpu-code2wav": gpu_code2wav,
            "--gpu-image-encoder": gpu_image_encoder,
            "--gpu-audio-encoder": gpu_audio_encoder,
        }
        if len(set(colocated_gpus.values())) != 1:
            raise ValueError(
                "--colocated requires all GPU stage flags to use the same GPU, "
                f"got {colocated_gpus}"
            )

    gpu_code_predictor = args.gpu_code_predictor
    if gpu_code_predictor is None:
        gpu_code_predictor = gpu_talker
    if gpu_code_predictor != gpu_talker:
        raise ValueError(
            "Qwen3 speech pipeline does not expose a separate code_predictor "
            "stage. Use the same GPU for --gpu-code-predictor and --gpu-talker."
        )

    config_cls = (
        Qwen3OmniSpeechColocatedPipelineConfig
        if args.colocated
        else Qwen3OmniSpeechPipelineConfig
    )
    config = config_cls(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    _set_stage_gpu(config, "image_encoder", gpu_image_encoder)
    _set_stage_gpu(config, "audio_encoder", gpu_audio_encoder)

    if args.thinker_tp_size < 1:
        raise ValueError(f"--thinker-tp-size must be >= 1, got {args.thinker_tp_size}")
    if args.thinker_tp_size > 1:
        if args.gpu_thinker_tp is None:
            raise ValueError(
                "--thinker-tp-size > 1 requires --gpu-thinker-tp "
                "(comma-separated GPU ids, one per TP rank)."
            )
        thinker_gpus = _parse_thinker_tp_gpu_list(
            args.gpu_thinker_tp,
            args.thinker_tp_size,
        )
        _set_stage_tp_size(config, "thinker", args.thinker_tp_size)
        _set_stage_gpu(config, "thinker", thinker_gpus)
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={
                "disable_custom_all_reduce": should_disable_custom_all_reduce_for_gpus(
                    thinker_gpus
                )
            },
        )
    else:
        if args.gpu_thinker_tp is not None:
            raise ValueError(
                "--gpu-thinker-tp only applies when --thinker-tp-size > 1; "
                "for TP=1, use --gpu-thinker."
            )
        _set_stage_gpu(config, "thinker", args.gpu_thinker)

    _set_stage_gpu(config, "talker_ar", gpu_talker)
    _set_stage_gpu(config, "code2wav", gpu_code2wav)
    _resolve_speech_mem_fractions(
        config,
        global_mem_fraction_static=args.mem_fraction_static,
        thinker_mem_fraction_static=args.thinker_mem_fraction_static,
        talker_mem_fraction_static=args.talker_mem_fraction_static,
    )
    if args.thinker_max_seq_len is not None:
        updates = {"thinker_max_seq_len": int(args.thinker_max_seq_len)}
        _apply_stage_factory_updates(config, stage_name="thinker", updates=updates)
        _apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=updates,
        )
    if args.talker_max_seq_len is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            updates={"talker_max_seq_len": int(args.talker_max_seq_len)},
        )
    partial_start_updates: dict[str, object] = {
        "enable_partial_start": enable_partial_start
    }
    if enable_partial_start:
        partial_start_updates["partial_start_min_chunks"] = int(
            args.partial_start_min_chunks
        )
    _apply_stage_factory_updates(
        config,
        stage_name="talker_ar",
        updates=partial_start_updates,
    )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def _build_ming_speech_server_parser() -> argparse.ArgumentParser:
    parser = _parser("Launch Ming-Omni with text and audio OpenAI responses.")
    _add_model_path(parser, "inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    _add_relay_backend(parser)
    parser.add_argument("--voice", type=str, default="DB30")
    _add_mem_fraction(parser, "Set mem_fraction_static for the thinker stage.")
    parser.add_argument("--cpu-offload-gb", type=int, default=None)
    parser.add_argument("--enable-streaming-tts", action="store_true")
    _add_server_args(parser, model_name="ming-omni")
    return parser


def launch_ming_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import (
        MingOmniSpeechPipelineConfig,
        MingOmniStreamingSpeechPipelineConfig,
    )
    from sglang_omni.serve import launch_server

    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    if args.enable_streaming_tts:
        config = MingOmniStreamingSpeechPipelineConfig(
            model_path=args.model_path,
            relay_backend=args.relay_backend,
        )
        talker_stage = "talker_stream"
        validate_gpus = config._validate_talker_stream_gpu_not_in_thinker_tp_range
    else:
        config = MingOmniSpeechPipelineConfig(
            model_path=args.model_path,
            relay_backend=args.relay_backend,
        )
        talker_stage = "talker"
        validate_gpus = config._validate_talker_gpu_not_in_thinker_tp_range

    if args.tp_size < 1:
        raise ValueError(f"--tp-size must be >= 1, got {args.tp_size}")
    _set_stage_tp_size(config, "thinker", args.tp_size)
    thinker_gpus: int | list[int] = int(args.gpu_thinker)
    if args.tp_size > 1:
        thinker_gpus = list(
            range(int(args.gpu_thinker), int(args.gpu_thinker) + int(args.tp_size))
        )
    _set_stage_gpu(config, "thinker", thinker_gpus)
    _set_stage_gpu(config, talker_stage, int(args.gpu_talker))
    validate_gpus()

    server_updates: dict[str, object] = {}
    if args.tp_size > 1:
        server_updates["disable_custom_all_reduce"] = True
    if args.mem_fraction_static is not None:
        server_updates["mem_fraction_static"] = args.mem_fraction_static
    if args.cpu_offload_gb is not None:
        server_updates["cpu_offload_gb"] = args.cpu_offload_gb
    if server_updates:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates=server_updates,
        )
    _apply_stage_factory_updates(
        config,
        stage_name=talker_stage,
        updates={"voice": args.voice},
    )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def _add_offline_args(
    parser: argparse.ArgumentParser,
    *,
    prompt: str,
    system: str,
    max_new_tokens: int,
    output: str | None,
) -> None:
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--system", type=str, default=system)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--temperature", type=float, default=0.7)
    _add_relay_backend(parser)
    parser.add_argument("--output", type=str, default=output)
    parser.add_argument("--timeout", type=float, default=300.0)


def _build_qwen_speech_parser() -> argparse.ArgumentParser:
    parser = _parser("Run one Qwen3-Omni text-to-speech request.")
    _add_model_path(parser, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    _add_offline_args(
        parser,
        prompt="Hello! Tell me something interesting.",
        system="You are a friendly assistant. Speak naturally and warmly.",
        max_new_tokens=64,
        output=None,
    )
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--gpu-code-predictor", type=int, default=None)
    parser.add_argument("--gpu-code2wav", type=int, default=0)
    parser.add_argument("--gpu-image-encoder", type=int, default=0)
    parser.add_argument("--gpu-audio-encoder", type=int, default=0)
    _add_qwen_speech_mem_args(parser)
    return parser


def _build_ming_speech_parser() -> argparse.ArgumentParser:
    parser = _parser("Run one Ming-Omni text-to-speech request.")
    _add_model_path(parser, "inclusionAI/Ming-flash-omni-2.0")
    _add_offline_args(
        parser,
        prompt="你好！给我讲一个有趣的事情。",
        system="你是一个友好的AI助手。请用自然、温暖的语气说话。",
        max_new_tokens=256,
        output="./output_audio.wav",
    )
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--voice", type=str, default="DB30")
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    _add_mem_fraction(parser, "Set mem_fraction_static for the thinker stage.")
    parser.add_argument("--tp-size", type=int, default=1)
    return parser


def _save_audio(result: dict, output_path: str) -> None:
    import numpy as np
    import torch

    for payload in result.values():
        data = payload if isinstance(payload, dict) else payload.data
        if not isinstance(data, dict):
            continue
        waveform = data.get("audio_waveform")
        if waveform is None:
            continue
        if isinstance(waveform, bytes):
            dtype = np.dtype(data.get("audio_waveform_dtype", "float32"))
            shape = data.get("audio_waveform_shape", [-1])
            waveform = np.frombuffer(waveform, dtype=dtype).reshape(shape)
        elif isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().float().numpy()
        waveform = waveform.squeeze()
        sample_rate = data.get("sample_rate", 24000)
        peak = max(abs(waveform.max()), abs(waveform.min()), 1e-8)
        waveform_int16 = (waveform / peak * 32767).astype(np.int16)
        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(waveform_int16.tobytes())
        logger.info(
            "Audio saved: %s (%.2fs, %d Hz)",
            output_path,
            len(waveform_int16) / sample_rate,
            sample_rate,
        )
        return
    logger.warning("No audio waveform found in pipeline result")


async def _run_speech_request(
    config: Any,
    *,
    request: dict[str, object],
    max_new_tokens: int,
    temperature: float,
    timeout: float,
    output: str | None,
    label: str,
) -> None:
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting %s speech pipeline...", label)
    await runner.start(timeout=600)
    try:
        started = time.time()
        result = await asyncio.wait_for(
            runner.coordinator.submit(
                "speech-request",
                OmniRequest(
                    inputs=request,
                    params={
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                ),
            ),
            timeout=timeout,
        )
        logger.info("Pipeline completed in %.2fs", time.time() - started)
        if output and isinstance(result, dict):
            _save_audio(result, output)
    finally:
        await runner.stop()


async def run_qwen_speech(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

    config = Qwen3OmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    if args.gpu_code_predictor not in (None, args.gpu_talker):
        raise ValueError(
            "Qwen3 speech pipeline does not expose a separate code_predictor "
            "stage. Use the same GPU for --gpu-code-predictor and --gpu-talker."
        )
    for stage_name, gpu_id in (
        ("thinker", args.gpu_thinker),
        ("talker_ar", args.gpu_talker),
        ("code2wav", args.gpu_code2wav),
        ("image_encoder", args.gpu_image_encoder),
        ("audio_encoder", args.gpu_audio_encoder),
    ):
        _set_stage_gpu(config, stage_name, gpu_id)
    _resolve_speech_mem_fractions(
        config,
        global_mem_fraction_static=args.mem_fraction_static,
        thinker_mem_fraction_static=args.thinker_mem_fraction_static,
        talker_mem_fraction_static=args.talker_mem_fraction_static,
    )
    request = {
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        "images": [],
        "videos": [],
        "audios": [],
    }
    await _run_speech_request(
        config,
        request=request,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        output=args.output,
        label="Qwen3-Omni",
    )


async def run_ming_speech(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    config = MingOmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    if args.tp_size < 1:
        raise ValueError(f"--tp-size must be >= 1, got {args.tp_size}")
    _set_stage_tp_size(config, "thinker", args.tp_size)
    thinker_gpus: int | list[int] = args.gpu_thinker
    if args.tp_size > 1:
        thinker_gpus = list(
            range(args.gpu_thinker, args.gpu_thinker + int(args.tp_size))
        )
    _set_stage_gpu(config, "thinker", thinker_gpus)
    _set_stage_gpu(config, "talker", args.gpu_talker)
    config._validate_talker_gpu_not_in_thinker_tp_range()
    overrides: dict[str, object] = {}
    if args.tp_size > 1:
        overrides["disable_custom_all_reduce"] = True
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb
    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    if args.mem_fraction_static is not None:
        overrides["mem_fraction_static"] = args.mem_fraction_static
    if overrides:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates=overrides,
        )

    content: object = args.prompt
    if args.audio_path:
        content = [
            {"type": "audio_url", "audio_url": {"url": args.audio_path}},
            {"type": "text", "text": args.prompt},
        ]
    request = {
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": content},
        ],
        "audios": [args.audio_path] if args.audio_path else [],
    }
    await _run_speech_request(
        config,
        request=request,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        output=args.output,
        label="Ming-Omni",
    )


def _build_ming_text_parser() -> argparse.ArgumentParser:
    parser = _parser("Run one Ming-Omni request with text output.")
    _add_model_path(parser, "inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--cpu-offload-gb", type=int, default=80)
    _add_mem_fraction(parser, "Set mem_fraction_static for the thinker stage.")
    _add_relay_backend(parser)
    return parser


async def run_ming_text(args: argparse.Namespace) -> None:
    from sglang_omni.config import build_pipeline_runner
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
    from sglang_omni.proto import OmniRequest

    config = MingOmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    overrides: dict[str, object] = {}
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb
    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    if args.mem_fraction_static is not None:
        overrides["mem_fraction_static"] = args.mem_fraction_static
    if overrides:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates=overrides,
        )

    content: object = args.prompt
    if args.audio_path:
        content = [
            {"type": "audio_url", "audio_url": {"url": args.audio_path}},
            {"type": "text", "text": args.prompt},
        ]
    runner = build_pipeline_runner(config)
    await runner.start()
    try:
        result = await runner.coordinator.submit(
            "ming-omni-text-first",
            OmniRequest(
                inputs={
                    "messages": [{"role": "user", "content": content}],
                    "audios": [args.audio_path] if args.audio_path else [],
                },
                params={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
            ),
        )
        print(result)
    finally:
        await runner.stop()


def _run_async(
    handler: Callable[[argparse.Namespace], Any], args: argparse.Namespace
) -> None:
    asyncio.run(handler(args))


@dataclass(frozen=True)
class LauncherPreset:
    description: str
    build_parser: Callable[[], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace], None]
    spawn: bool = False


PRESETS = {
    "qwen3-text-server": LauncherPreset(
        "Qwen3-Omni text server",
        _build_qwen_text_server_parser,
        launch_qwen_text_server,
    ),
    "qwen3-speech-server": LauncherPreset(
        "Qwen3-Omni speech server",
        _build_qwen_speech_server_parser,
        launch_qwen_speech_server,
        spawn=True,
    ),
    "qwen3-speech": LauncherPreset(
        "One Qwen3-Omni speech request",
        _build_qwen_speech_parser,
        lambda args: _run_async(run_qwen_speech, args),
        spawn=True,
    ),
    "ming-text-server": LauncherPreset(
        "Ming-Omni text server",
        _build_ming_text_server_parser,
        launch_ming_text_server,
        spawn=True,
    ),
    "ming-speech-server": LauncherPreset(
        "Ming-Omni speech server",
        _build_ming_speech_server_parser,
        launch_ming_speech_server,
        spawn=True,
    ),
    "ming-speech": LauncherPreset(
        "One Ming-Omni speech request",
        _build_ming_speech_parser,
        lambda args: _run_async(run_ming_speech, args),
        spawn=True,
    ),
    "ming-text": LauncherPreset(
        "One Ming-Omni text request",
        _build_ming_text_parser,
        lambda args: _run_async(run_ming_text, args),
    ),
}


def parse_preset_args(
    preset_name: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return PRESETS[preset_name].build_parser().parse_args(argv)


def run_preset(
    preset_name: str,
    argv: Sequence[str] | None = None,
) -> None:
    preset = PRESETS[preset_name]
    if preset.spawn:
        mp.set_start_method("spawn", force=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    preset.run(parse_preset_args(preset_name, argv))


def run_cli(argv: Sequence[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] in PRESETS:
        run_preset(argv[0], argv[1:])
        return
    selector = argparse.ArgumentParser(
        description="Run an Omni example through a reusable launcher preset."
    )
    selector.add_argument("preset", choices=sorted(PRESETS))
    selector.parse_args(argv)
