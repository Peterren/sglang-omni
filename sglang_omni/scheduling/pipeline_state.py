# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for per-request state carried between pipeline stages."""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING, dataclass, field
from typing import Any, Callable, TypeVar

from sglang_omni.proto import StagePayload

StateT = TypeVar("StateT", bound="PipelineStateBase")

__all__ = [
    "DeclarativeStateBase",
    "PipelineStateBase",
    "build_usage",
    "load_state",
    "store_state",
    "wire",
]

_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "engine_time_s")
_EXPLICIT_EMIT_MODES = frozenset({"always", "not_none", "truthy"})


@dataclass
class PipelineStateBase:
    """Shared usage/serialization mechanics; tensor strategy stays subclass-owned."""

    sample_rate: int = 24000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # Note(Chenchen Hong): subclasses must override; the stub turns a forgotten
    # override into a clear contract error rather than an AttributeError in store_state.
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} must implement to_dict()")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStateBase":
        raise NotImplementedError(f"{cls.__name__} must implement from_dict()")

    @staticmethod
    def serialize_value(value: Any) -> Any:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def append_usage_fields(self, data: dict[str, Any]) -> None:
        if self.prompt_tokens:
            data["prompt_tokens"] = int(self.prompt_tokens)
        if self.completion_tokens:
            data["completion_tokens"] = int(self.completion_tokens)
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)


def _tensor_to_list(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _tensor_from_list(value: Any, _default: Any = None) -> Any:
    if value is None:
        return None
    import torch

    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def _tensor_items_to_lists(value: Any) -> Any:
    return [_tensor_to_list(item) for item in value]


def _tensor_items_from_lists(value: Any, _default: Any = None) -> Any:
    if value is None:
        return None
    return [_tensor_from_list(item) for item in value]


# note (luojiaxuan): Wire codecs are (encode, decode). Encode runs on the
# field value at to_dict time after the emit rule admits it; decode runs at
# from_dict time only when the key is present in the payload, so absent keys
# fall back to the dataclass default. Decode receives the field default for
# star_or variants that treat falsy wire values as "use the default".
_CODECS: dict[str, tuple[Callable[[Any], Any], Callable[[Any, Any], Any]]] = {
    "raw": (lambda v: v, lambda v, d: v),
    "int": (int, lambda v, d: int(v or 0)),
    "int_or": (int, lambda v, d: int(v or d)),
    "opt_int": (int, lambda v, d: int(v) if v is not None else None),
    "float": (float, lambda v, d: float(v or 0.0)),
    "str": (str, lambda v, d: str(v)),
    "str_or": (str, lambda v, d: str(v or d)),
    "bool": (bool, lambda v, d: bool(v)),
    "dict": (dict, lambda v, d: dict(v) if isinstance(v, dict) else {}),
    "list": (list, lambda v, d: list(v) if v is not None else None),
    # note (luojiaxuan): Tensor stays native on the wire because payload dicts
    # stay in-process and the relay handles tensor transport; detach and move it
    # to CPU before storing.
    "tensor_cpu": (PipelineStateBase.serialize_value, lambda v, d: v),
    # note (luojiaxuan): Tensor flattens to nested lists and stays a list after restore.
    "tensor_list": (_tensor_to_list, lambda v, d: v),
    # note (luojiaxuan): Tensor flattens to nested lists and restores back to a tensor.
    "tensor_restore": (_tensor_to_list, _tensor_from_list),
    # note (luojiaxuan): Lists of tensors flatten and restore element-wise.
    "tensor_items": (_tensor_items_to_lists, _tensor_items_from_lists),
}


@dataclass(frozen=True)
class _WireSpec:
    emit: str | None = None  # always | not_none | truthy | with:<field>
    codec: str = "raw"


_DEFAULT_SPEC = _WireSpec()


@dataclass(frozen=True)
class _FieldPlan:
    name: str
    emit: str
    codec: str
    anchor: str | None
    encode: Callable[[Any], Any] | None
    decode: Callable[[Any, Any], Any] | None
    default_value: Any
    default_factory: Callable[[], Any] | None
    is_usage: bool = False

    def default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        return self.default_value


_PLAN_CACHE: dict[type[Any], tuple[_FieldPlan, ...]] = {}


def _validate_emit_mode(emit: str | None) -> None:
    if emit is None or emit in _EXPLICIT_EMIT_MODES:
        return
    if emit.startswith("with:"):
        if not emit.split(":", 1)[1]:
            raise ValueError("wire emit with anchor must not be empty")
        return
    raise ValueError(f"unknown wire emit mode: {emit!r}")


def wire(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
    emit: str | None = None,
    codec: str = "raw",
) -> Any:
    """dataclasses.field carrying wire metadata for DeclarativeStateBase.

    emit defaults by inference: fields whose default is None emit only when
    not None; everything else always emits. codec="typed_tensor" expands to
    the {name}_bytes/_shape/_dtype key triple via scheduling.typed_tensor.
    """
    _validate_emit_mode(emit)
    if codec != "typed_tensor" and codec not in _CODECS:
        raise ValueError(f"unknown wire codec: {codec!r}")
    metadata = {"wire": _WireSpec(emit=emit, codec=codec)}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def _spec_of(f: dataclasses.Field) -> _WireSpec:
    return f.metadata.get("wire", _DEFAULT_SPEC)


def _default_of(f: dataclasses.Field) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def _emit_kind(f: dataclasses.Field, spec: _WireSpec) -> str:
    _validate_emit_mode(spec.emit)
    if spec.emit is not None:
        return spec.emit
    if f.default is not MISSING and f.default is None:
        return "not_none"
    return "always"


def _plan_default_factory(f: dataclasses.Field) -> Callable[[], Any] | None:
    if f.default_factory is MISSING:  # type: ignore[misc]
        return None
    return f.default_factory  # type: ignore[return-value,misc]


def _has_encoded_typed_tensor_payload(data: dict[str, Any], name: str) -> bool:
    return (
        f"{name}_bytes" in data
        or f"{name}_shape" in data
        or f"{name}_dtype" in data
    )


def _has_typed_tensor_payload(data: dict[str, Any], name: str) -> bool:
    return name in data or _has_encoded_typed_tensor_payload(data, name)


def _build_plan(cls: type[Any]) -> tuple[_FieldPlan, ...]:
    fields = dataclasses.fields(cls)
    field_names = {f.name for f in fields}
    plans: list[_FieldPlan] = []
    for f in fields:
        spec = _spec_of(f)
        emit = _emit_kind(f, spec)
        anchor = emit.split(":", 1)[1] if emit.startswith("with:") else None
        if anchor is not None and anchor not in field_names:
            raise ValueError(
                f"wire emit anchor {anchor!r} for {cls.__name__}.{f.name} "
                "is not a dataclass field"
            )
        encode: Callable[[Any], Any] | None
        decode: Callable[[Any, Any], Any] | None
        if spec.codec == "typed_tensor":
            encode = None
            decode = None
        else:
            encode, decode = _CODECS[spec.codec]
        plans.append(
            _FieldPlan(
                name=f.name,
                emit=emit,
                codec=spec.codec,
                anchor=anchor,
                encode=encode,
                decode=decode,
                default_value=_default_of(f),
                default_factory=_plan_default_factory(f),
                is_usage=f.name in _USAGE_FIELDS,
            )
        )
    return tuple(plans)


def _plan_for(cls: type[Any]) -> tuple[_FieldPlan, ...]:
    plan = _PLAN_CACHE.get(cls)
    if plan is None:
        plan = _build_plan(cls)
        _PLAN_CACHE[cls] = plan
    return plan


@dataclass
class DeclarativeStateBase(PipelineStateBase):
    """PipelineStateBase with to_dict/from_dict derived from field metadata.

    Subclasses declare wire behavior inline with wire(...) fields instead of
    hand-writing the serialization pair; plain fields default to
    always-emitted raw passthrough (None-defaulted fields emit only when set).
    Usage fields keep the append_usage_fields contract. The field-complete
    round-trip contract test in tests/unit_test/scheduling/test_pipeline_state.py
    pins both the wire layout and the restored attributes per model.
    """

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        coupled: list[_FieldPlan] = []
        for plan in _plan_for(type(self)):
            if plan.is_usage:
                continue
            if plan.anchor is not None:
                coupled.append(plan)
                continue
            self._encode_field(data, plan, plan.emit)
        for plan in coupled:
            if plan.anchor in data:
                self._encode_field(data, plan, "always")
        self.append_usage_fields(data)
        return data

    def _encode_field(
        self,
        data: dict[str, Any],
        plan: _FieldPlan,
        emit: str,
    ) -> None:
        value = getattr(self, plan.name)
        if emit == "not_none" and value is None:
            return
        if emit == "truthy" and not value:
            return
        if plan.codec == "typed_tensor":
            if value is not None:
                from sglang_omni.scheduling.typed_tensor import encode_typed_tensor

                data.update(encode_typed_tensor(value, key=plan.name))
            return
        assert plan.encode is not None
        encode = plan.encode
        data[plan.name] = encode(value)

    @classmethod
    def from_dict(cls: type[StateT], data: Any) -> StateT:
        if not isinstance(data, dict):
            data = {}
        kwargs: dict[str, Any] = {}
        for plan in _plan_for(cls):
            if plan.codec == "typed_tensor":
                if not _has_typed_tensor_payload(data, plan.name):
                    continue
                if (
                    plan.name in data
                    and data[plan.name] is None
                    and not _has_encoded_typed_tensor_payload(data, plan.name)
                ):
                    kwargs[plan.name] = None
                    continue
                from sglang_omni.scheduling.typed_tensor import decode_typed_tensor

                kwargs[plan.name] = decode_typed_tensor(
                    data, key=plan.name, legacy_key=plan.name
                )
                continue
            if plan.name == "prompt_tokens":
                kwargs[plan.name] = int(data.get("prompt_tokens", 0) or 0)
                continue
            if plan.name == "completion_tokens":
                kwargs[plan.name] = int(data.get("completion_tokens", 0) or 0)
                continue
            if plan.name == "engine_time_s":
                kwargs[plan.name] = float(data.get("engine_time_s", 0.0) or 0.0)
                continue
            if plan.name not in data:
                continue
            assert plan.decode is not None
            kwargs[plan.name] = plan.decode(data[plan.name], plan.default())
        return cls(**kwargs)


def load_state(payload: StagePayload, state_cls: type[StateT]) -> StateT:
    return state_cls.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineStateBase) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def build_usage(state: PipelineStateBase) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage
