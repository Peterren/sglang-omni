# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def test_tts_engine_builder_import_is_cpu_only() -> None:
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    assert TtsEngineBuilder.__name__ == "TtsEngineBuilder"


def test_tts_engine_builder_phase_order_and_override_contract(monkeypatch) -> None:
    from sglang_omni.scheduling import bootstrap, sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    events: list[str] = []
    build_kwargs: dict[str, Any] = {}
    infrastructure_saw_graph_disabled: list[bool] = []
    init_graph_calls: list[bool] = []

    class FakeModel:
        pass

    class FakeSGLangRunner:
        def __init__(self, server_args: Any) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_device_graphs(self) -> None:
            events.append("init_graphs")
            init_graph_calls.append(True)

    class FakeWorker:
        def __init__(self, server_args: Any) -> None:
            self.model_runner = FakeSGLangRunner(server_args)

    def fake_build_sglang_server_args(
        checkpoint_dir: str,
        *,
        context_length: int,
        **kwargs: Any,
    ) -> Any:
        events.append("build_server_args")
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            context_length=context_length,
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            enable_torch_compile=kwargs["enable_torch_compile"],
            max_running_requests=kwargs["max_running_requests"],
            mem_fraction_static=kwargs["mem_fraction_static"],
            torch_compile_max_bs=kwargs["torch_compile_max_bs"],
        )

    def fake_create_sglang_infrastructure(
        server_args: Any,
        gpu_id: int,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        events.append("infrastructure")
        assert gpu_id == 2
        assert kwargs == {"model_arch_override": "TestArch"}
        infrastructure_saw_graph_disabled.append(bool(server_args.disable_cuda_graph))
        return (
            FakeWorker(server_args),
            "tree_cache",
            "req_pool",
            "kv_pool",
            "prefill",
            "decode",
            "model_config",
        )

    def fake_output_processor(**kwargs: Any) -> Any:
        events.append("output_processor")
        assert kwargs["capture_hidden"] is False
        assert kwargs["capture_hidden_layers"] is None
        assert isinstance(kwargs["model"], FakeModel)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", fake_output_processor)

    class RecordingBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123
        model_arch_override = "TestArch"

        def resolve_checkpoint(self, model_path: str) -> str:
            events.append("resolve_checkpoint")
            return f"{model_path}-resolved"

        def pre_infra_setup(self, checkpoint_dir: str) -> None:
            events.append("pre_infra_setup")
            assert checkpoint_dir == "model-resolved"

        def generation_defaults(
            self,
            *,
            dtype: str,
            server_args_overrides: dict[str, Any] | None,
            **model_kwargs: Any,
        ) -> dict[str, Any]:
            events.append("generation_defaults")
            assert dtype == "bfloat16"
            assert server_args_overrides is not None
            assert model_kwargs == {}
            return {
                "max_running_requests": 4,
                "cuda_graph_max_bs": 4,
                "torch_compile_max_bs": 4,
                "dtype": dtype,
                "disable_cuda_graph": False,
                "enable_torch_compile": True,
                "mem_fraction_static": 0.5,
            }

        def adjust_overrides(
            self,
            overrides: dict[str, Any],
            **model_kwargs: Any,
        ) -> None:
            del model_kwargs
            events.append("adjust_overrides")
            assert overrides["mem_fraction_static"] == 0.7

        def customize_server_args(self, server_args: Any) -> None:
            events.append("customize_server_args")
            assert server_args.context_length == 123

        def setup_model(
            self,
            *,
            model_worker: Any,
            checkpoint_dir: str,
            device: str,
            gpu_id: int,
            server_args: Any,
        ) -> None:
            events.append("setup_model")
            assert isinstance(model_worker.model_runner.model, FakeModel)
            assert checkpoint_dir == "model-resolved"
            assert device == "cuda:2"
            assert gpu_id == 2
            assert server_args.disable_cuda_graph is False

        def get_model_buffer_bs(self, model: Any) -> int | None:
            events.append("get_model_buffer_bs")
            assert isinstance(model, FakeModel)
            return 2

        def compile_model(self, model: Any, server_args: Any) -> None:
            events.append("compile_model")
            assert isinstance(model, FakeModel)
            assert server_args.torch_compile_max_bs == 8

        def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
            events.append("post_cuda_graph_setup")
            assert isinstance(model, FakeModel)
            assert server_args.disable_cuda_graph is False

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            events.append("make_model_runner")
            return SimpleNamespace(model_worker=model_worker, output_proc=output_proc)

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            events.append("make_adapters")
            assert isinstance(model, FakeModel)
            return "request_builder", "result_adapter"

        def make_scheduler(self, **kwargs: Any) -> Any:
            events.append("make_scheduler")
            assert kwargs["request_builder"] == "request_builder"
            assert kwargs["result_adapter"] == "result_adapter"
            return SimpleNamespace(outbox="outbox", kwargs=kwargs)

        def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
            events.append("post_scheduler_setup")
            model_runner.outbox = scheduler.outbox

    scheduler = RecordingBuilder().build(
        "model",
        device="cuda:0",
        gpu_id=2,
        server_args_overrides={
            "cuda_graph_max_bs": 8,
            "torch_compile_max_bs": 8,
            "mem_fraction_static": 0.7,
            "max_running_requests": 2,
        },
    )

    assert events == [
        "resolve_checkpoint",
        "pre_infra_setup",
        "generation_defaults",
        "adjust_overrides",
        "build_server_args",
        "customize_server_args",
        "infrastructure",
        "setup_model",
        "get_model_buffer_bs",
        "compile_model",
        "init_graphs",
        "post_cuda_graph_setup",
        "output_processor",
        "make_model_runner",
        "make_adapters",
        "make_scheduler",
        "post_scheduler_setup",
    ]
    assert build_kwargs["max_running_requests"] == 2
    assert build_kwargs["cuda_graph_max_bs"] == 8
    assert build_kwargs["torch_compile_max_bs"] == 8
    assert build_kwargs["mem_fraction_static"] == 0.7
    assert infrastructure_saw_graph_disabled == [True]
    assert init_graph_calls == [True]
    assert scheduler.kwargs["server_args"].disable_cuda_graph is False
    assert scheduler.kwargs["model_runner"].outbox == "outbox"
