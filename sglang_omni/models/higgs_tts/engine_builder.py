# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS SGLang engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts import stages as higgs_stages
from sglang_omni.models.higgs_tts import utils as higgs_utils
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class HiggsTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Higgs TTS"
    context_length = 4096
    # Largest token bucket captured for piecewise prefill graphs. Higgs
    # voice-clone prompts (ref-audio codes + text) stay well under this, so
    # 2048 covers real prefill chunks without paying capture time for buckets
    # the workload never hits.
    piecewise_cuda_graph_max_tokens = 2048

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
        enable_piecewise_cuda_graph: bool = False,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.enable_piecewise_cuda_graph = enable_piecewise_cuda_graph
        self.model: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return higgs_stages.resolve_checkpoint(model_path)

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        del dtype
        # note (luojiaxuan): Radix cache is namespaced per ref-audio via
        # Req.extra_key (set in build_sglang_higgs_request); shared -100
        # placeholder prefixes from different ref audios can't cross-contaminate
        # the KV tree.
        return {
            "max_running_requests": self.max_running_requests,
            "cuda_graph_max_bs": self.cuda_graph_max_bs,
            "disable_cuda_graph": False,
            "mem_fraction_static": 0.85,
            "chunked_prefill_size": 8192,
            "dtype": "bfloat16",
        }

    def customize_server_args(self, server_args: Any) -> None:
        server_args.disable_overlap_schedule = True
        if self.enable_piecewise_cuda_graph:
            # ServerArgs.__post_init__ has already run at this point and its
            # blanket multimodal gate turned piecewise CUDA graph off (and left
            # the token buckets ungenerated). Higgs does audio-code embedding
            # outside the Qwen3 backbone forward, so the LM prefill path is
            # safe to capture: re-enable PCG and rebuild the buckets here.
            server_args.enforce_piecewise_cuda_graph = True
            server_args.disable_piecewise_cuda_graph = False
            server_args.piecewise_cuda_graph_max_tokens = (
                self.piecewise_cuda_graph_max_tokens
            )
            server_args.piecewise_cuda_graph_tokens = (
                server_args._generate_piecewise_cuda_graph_tokens()
            )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        self.model = model_worker.model_runner.model
        higgs_utils.truncate_rope_to_bf16(self.model)

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return model.sampler_pool_max_running_requests

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.higgs_tts.model_runner"
        )

        return model_runner_mod.HiggsTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_higgs_scheduler_adapters(
            model,
            max_new_tokens_cap=self.max_new_tokens,
        )

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        return self.model.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
