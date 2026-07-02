# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local SGLang engine builder."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.moss_tts_local import request_builders
from sglang_omni.models.moss_tts_local import stages as moss_local_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class MossTtsLocalEngineBuilder(TtsEngineBuilder):
    model_name = "MOSS-TTS Local"
    context_length = 8192
    model_arch_override = "MossTTSLocalSGLangModel"

    def __init__(
        self,
        *,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        total_gpu_memory_fraction: float | None,
        codec_mem_reserve: float,
    ) -> None:
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.codec_mem_reserve = codec_mem_reserve
        self.memory_budget = moss_local_stages._ArMemoryBudget(
            effective_total_gpu_memory_fraction=None,
            applied_codec_mem_reserve=0.0,
        )
        self.profile_total_gpu_memory_fraction: float | None = None
        self.model: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return moss_local_stages.resolve_moss_checkpoint(model_path)

    def generation_defaults(
        self,
        *,
        dtype: str,
        server_args_overrides: dict[str, Any] | None,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        del server_args_overrides, model_kwargs
        defaults: dict[str, Any] = {
            "max_running_requests": 16,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": 8192,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }
        if self.total_gpu_memory_fraction is None:
            defaults["mem_fraction_static"] = (
                0.6 if moss_local_stages.torch.cuda.device_count() > 1 else 0.5
            )
        return defaults

    def adjust_overrides(self, overrides: dict[str, Any], **model_kwargs: Any) -> None:
        del model_kwargs
        self.memory_budget = moss_local_stages._apply_colocated_ar_memory_budget(
            overrides,
            total_gpu_memory_fraction=self.total_gpu_memory_fraction,
            codec_mem_reserve=self.codec_mem_reserve,
        )
        self.profile_total_gpu_memory_fraction = (
            self.memory_budget.effective_total_gpu_memory_fraction
        )
        if self.profile_total_gpu_memory_fraction is None:
            return

        from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

        if get_process_gpu_memory_bytes(self.gpu_id) is None:
            moss_local_stages.logger.warning(
                f"MOSS-TTS Local colocated process memory accounting is unavailable; "
                f"falling back to upstream SGLang free-memory profiling. "
                f"effective_total_gpu_memory_fraction="
                f"{self.profile_total_gpu_memory_fraction}"
            )
            self.profile_total_gpu_memory_fraction = None

    def customize_server_args(self, server_args: Any) -> None:
        moss_local_stages.logger.info(
            f"MOSS-TTS Local SGLang startup: gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self.total_gpu_memory_fraction} "
            f"effective_total_gpu_memory_fraction="
            f"{self.memory_budget.effective_total_gpu_memory_fraction} "
            f"codec_mem_reserve={self.memory_budget.applied_codec_mem_reserve:.3f} "
            f"mem_fraction_static={server_args.mem_fraction_static} "
            f"profile_total_gpu_memory_fraction="
            f"{self.profile_total_gpu_memory_fraction}"
        )

    def infra_kwargs(self) -> dict[str, Any]:
        return {
            "total_gpu_memory_fraction": self.profile_total_gpu_memory_fraction,
        }

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

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        model.init_frame_decode_graphs(list(server_args.cuda_graph_bs))

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.moss_tts_local import model_runner as model_runner_mod

        return model_runner_mod.MossTTSLocalModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_moss_tts_local_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        def abort_request(request_id: str) -> None:
            request_builders.cleanup_prepared_moss_tts_local_request(request_id)
            if self.model is not None:
                self.model.reset_request(request_id)

        return abort_request

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
