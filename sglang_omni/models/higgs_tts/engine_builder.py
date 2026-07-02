# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS SGLang engine builder."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.higgs_tts import model_runner as model_runner_mod
from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts import stages as higgs_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class HiggsTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Higgs TTS"
    context_length = 4096

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.model: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return higgs_stages.resolve_checkpoint(model_path)

    def generation_defaults(
        self,
        *,
        dtype: str,
        server_args_overrides: dict[str, Any] | None,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        del dtype, server_args_overrides, model_kwargs
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
        higgs_stages.truncate_rope_to_bf16(self.model)

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return model.sampler_pool_max_running_requests

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        return model_runner_mod.HiggsTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_higgs_scheduler_adapters(
            model,
            max_new_tokens_cap=self.max_new_tokens,
        )

    def make_abort_callback(self) -> Any | None:
        if self.model is None:
            return None
        return self.model.reset_request

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        return omni_scheduler.OmniScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            abort_callback=self.make_abort_callback(),
            enable_async_decode=self.enable_async_decode,
            async_decode_min_batch_size=self.async_decode_min_batch_size,
        )

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
