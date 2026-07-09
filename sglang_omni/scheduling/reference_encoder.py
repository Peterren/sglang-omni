# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
import json
import logging
import queue as _queue_mod
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar

from sglang_omni.profiler.event_recorder import emit as _emit_profiler_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
ArtifactT = TypeVar("ArtifactT")
StoredT = TypeVar("StoredT")


@dataclass(frozen=True)
class ReferenceEncodeKey:
    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str
    input_key: str
    options_key: str = ""

    def to_string(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class ReferenceEncodeHook(Generic[InputT, ArtifactT, StoredT]):
    def normalize_input(self, raw_input: Any) -> InputT:
        raise NotImplementedError

    def cache_key(self, item: InputT) -> ReferenceEncodeKey | None:
        raise NotImplementedError

    def encode_one(self, item: InputT) -> ArtifactT:
        raise NotImplementedError

    def store_artifact(self, artifact: ArtifactT) -> StoredT:
        raise NotImplementedError

    def load_artifact(self, stored: StoredT) -> ArtifactT:
        raise NotImplementedError

    def revalidate(self, item: InputT, key: ReferenceEncodeKey) -> bool:
        return True

    def can_encode_batch(self) -> bool:
        return False

    def encode_batch(self, items: list[InputT]) -> list[ArtifactT]:
        return [self.encode_one(item) for item in items]


def _fresh_exception(exc: BaseException) -> BaseException:
    try:
        fresh = type(exc)(*getattr(exc, "args", ()))
    except Exception:
        fresh = RuntimeError(str(exc))
    for note in getattr(exc, "__notes__", ()):
        add_note = getattr(fresh, "add_note", None)
        if callable(add_note):
            add_note(note)
    return fresh


@dataclass
class _BatchJob(Generic[InputT, StoredT]):
    item: InputT
    key: ReferenceEncodeKey
    cache_key: str
    future: concurrent.futures.Future[StoredT]
    desc: str | None
    request_id: str | None


class ReferenceEncodeService(Generic[InputT, ArtifactT, StoredT]):
    _LOG_INTERVAL_S = 60.0
    _BATCH_STOP = object()

    def __init__(
        self,
        hook: ReferenceEncodeHook[InputT, ArtifactT, StoredT],
        *,
        max_items: int | None = 256,
        max_bytes: int | None = 64 * 1024 * 1024,
        timeout_s: float = 130.0,
        log_prefix: str | None = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
    ) -> None:
        if max_items is not None and max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_batch_wait_ms < 0:
            raise ValueError(
                f"max_batch_wait_ms must be >= 0, got {max_batch_wait_ms}"
            )
        self._hook = hook
        self._cache = StageOutputCache(max_size=max_items, max_bytes=max_bytes)
        self._timeout_s = float(timeout_s)
        self._log_prefix = log_prefix
        self._max_batch_size = int(max_batch_size)
        self._max_batch_wait_s = float(max_batch_wait_ms) / 1000.0
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future[StoredT]] = {}
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._failed = 0
        self._uncacheable = 0
        self._batches = 0
        self._batched_items = 0
        self._batch_fallbacks = 0
        self._last_log_time = 0.0
        self._batch_enabled = self._max_batch_size > 1 and hook.can_encode_batch()
        self._batch_queue: _queue_mod.Queue[Any] | None = None
        self._batch_thread: threading.Thread | None = None
        if self._batch_enabled:
            self._batch_queue = _queue_mod.Queue()
            self._batch_thread = threading.Thread(
                target=self._batch_worker,
                name="reference-encode-batch",
                daemon=True,
            )
            self._batch_thread.start()

    @staticmethod
    def _event_metadata(
        key: ReferenceEncodeKey | None,
        **extra: Any,
    ) -> dict[str, Any]:
        metadata = {name: value for name, value in extra.items() if value is not None}
        if key is not None:
            metadata.update(
                {
                    "model_id": key.model_id,
                    "encoder_id": key.encoder_id,
                    "artifact_kind": key.artifact_kind,
                }
            )
        return metadata

    @staticmethod
    def _emit_event(
        request_id: str | None,
        event_name: str,
        *,
        key: ReferenceEncodeKey | None = None,
        **metadata: Any,
    ) -> None:
        if request_id is None or not _get_event_recorder().is_active():
            return
        _emit_profiler_event(
            request_id=str(request_id),
            stage=None,
            event_name=event_name,
            metadata=ReferenceEncodeService._event_metadata(key, **metadata),
        )

    def get_or_encode(
        self,
        raw_input: Any,
        *,
        desc: str | None = None,
        request_id: str | None = None,
    ) -> ArtifactT:
        item = self._hook.normalize_input(raw_input)
        key = self._hook.cache_key(item)
        if key is None:
            with self._lock:
                self._uncacheable += 1
            self._emit_event(
                request_id,
                "reference_encode_lookup",
                result="uncacheable",
            )
            self._emit_event(
                request_id,
                "reference_encode_start",
                result="uncacheable",
            )
            try:
                artifact = self._hook.encode_one(item)
            except BaseException as exc:
                self._add_exception_note(exc, desc)
                with self._lock:
                    self._failed += 1
                self._emit_event(
                    request_id,
                    "reference_encode_end",
                    result="error",
                    error_type=type(exc).__name__,
                )
                self._emit_event(
                    request_id,
                    "reference_encode_failure",
                    result="error",
                    phase="encode",
                    cacheable=False,
                    error_type=type(exc).__name__,
                )
                raise
            self._emit_event(
                request_id,
                "reference_encode_end",
                result="success",
                cacheable=False,
            )
            return artifact

        cache_key = key.to_string()
        leader_fut: concurrent.futures.Future[StoredT] | None = None
        follower_fut: concurrent.futures.Future[StoredT] | None = None
        stored: StoredT | None = None
        lookup_result: str
        with self._lock:
            stored = self._cache.get(cache_key)
            if stored is not None:
                self._hits += 1
                lookup_result = "hit"
            elif cache_key in self._inflight:
                self._merged += 1
                follower_fut = self._inflight[cache_key]
                lookup_result = "merged"
            else:
                self._misses += 1
                leader_fut = concurrent.futures.Future()
                self._inflight[cache_key] = leader_fut
                lookup_result = "miss"

        self._emit_event(
            request_id,
            "reference_encode_lookup",
            key=key,
            result=lookup_result,
        )

        if stored is not None:
            self._maybe_log()
            return self._hook.load_artifact(stored)

        if follower_fut is not None:
            self._emit_event(
                request_id,
                "reference_encode_wait_start",
                key=key,
                result="merged",
            )
            try:
                stored = follower_fut.result(timeout=self._timeout_s)
            except concurrent.futures.TimeoutError as exc:
                self._add_exception_note(exc, desc)
                self._emit_event(
                    request_id,
                    "reference_encode_wait_end",
                    key=key,
                    result="timeout",
                )
                self._emit_event(
                    request_id,
                    "reference_encode_failure",
                    key=key,
                    result="timeout",
                    phase="wait",
                    error_type=type(exc).__name__,
                )
                raise
            except BaseException as exc:
                self._add_exception_note(exc, desc)
                self._emit_event(
                    request_id,
                    "reference_encode_wait_end",
                    key=key,
                    result="error",
                    error_type=type(exc).__name__,
                )
                self._emit_event(
                    request_id,
                    "reference_encode_failure",
                    key=key,
                    result="error",
                    phase="wait",
                    error_type=type(exc).__name__,
                )
                raise _fresh_exception(exc) from exc
            self._emit_event(
                request_id,
                "reference_encode_wait_end",
                key=key,
                result="success",
            )
            return self._hook.load_artifact(stored)

        assert leader_fut is not None
        if self._batch_enabled:
            return self._enqueue_batch_job(
                item=item,
                key=key,
                cache_key=cache_key,
                future=leader_fut,
                desc=desc,
                request_id=request_id,
            )

        # note (luojiaxuan): revalidate and cache put share the encode guard.
        # A failure must drop inflight and fail the future so same-key followers
        # do not wait on a dead leader after a reference mutates mid-encode.
        encode_event_closed = False
        try:
            self._emit_event(
                request_id,
                "reference_encode_start",
                key=key,
                result="miss",
            )
            artifact = self._hook.encode_one(item)
            self._emit_event(
                request_id,
                "reference_encode_end",
                key=key,
                result="success",
                cacheable=True,
            )
            encode_event_closed = True
            stored = self._hook.store_artifact(artifact)
            should_cache = self._hook.revalidate(item, key)
            with self._lock:
                if should_cache:
                    self._cache.put(cache_key, stored)
                self._inflight.pop(cache_key, None)
        except BaseException as exc:
            self._add_exception_note(exc, desc)
            with self._lock:
                self._inflight.pop(cache_key, None)
                self._failed += 1
            if not encode_event_closed:
                self._emit_event(
                    request_id,
                    "reference_encode_end",
                    key=key,
                    result="error",
                    error_type=type(exc).__name__,
                )
            self._emit_event(
                request_id,
                "reference_encode_failure",
                key=key,
                result="error",
                phase="post_encode" if encode_event_closed else "encode",
                error_type=type(exc).__name__,
            )
            leader_fut.set_exception(exc)
            raise
        leader_fut.set_result(stored)
        self._maybe_log()
        return self._hook.load_artifact(stored)

    def _enqueue_batch_job(
        self,
        *,
        item: InputT,
        key: ReferenceEncodeKey,
        cache_key: str,
        future: concurrent.futures.Future[StoredT],
        desc: str | None,
        request_id: str | None,
    ) -> ArtifactT:
        assert self._batch_queue is not None
        self._batch_queue.put(
            _BatchJob(
                item=item,
                key=key,
                cache_key=cache_key,
                future=future,
                desc=desc,
                request_id=request_id,
            )
        )
        try:
            stored = future.result(timeout=self._timeout_s)
        except BaseException as exc:
            self._add_exception_note(exc, desc)
            raise
        self._maybe_log()
        return self._hook.load_artifact(stored)

    def _batch_worker(self) -> None:
        assert self._batch_queue is not None
        while True:
            job = self._batch_queue.get()
            if job is self._BATCH_STOP:
                return
            batch = [job]
            deadline = time.monotonic() + self._max_batch_wait_s
            while len(batch) < self._max_batch_size:
                try:
                    if self._max_batch_wait_s <= 0:
                        next_job = self._batch_queue.get_nowait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        next_job = self._batch_queue.get(timeout=remaining)
                except _queue_mod.Empty:
                    break
                if next_job is self._BATCH_STOP:
                    self._batch_queue.put(self._BATCH_STOP)
                    break
                batch.append(next_job)
            self._run_batch_jobs(batch)

    def _run_batch_jobs(self, batch: list[_BatchJob[InputT, StoredT]]) -> None:
        for job in batch:
            self._emit_event(
                job.request_id,
                "reference_encode_start",
                key=job.key,
                result="miss",
            )
        try:
            artifacts = self._hook.encode_batch([job.item for job in batch])
            if len(artifacts) != len(batch):
                raise ValueError(
                    "encode_batch returned "
                    f"{len(artifacts)} results for {len(batch)} inputs"
                )
        except BaseException:
            logger.warning(
                "Reference encode batch failed; retrying each item",
                exc_info=True,
            )
            with self._lock:
                self._batch_fallbacks += 1
            for job in batch:
                self._run_batch_job_one(job)
            return

        with self._lock:
            self._batches += 1
            self._batched_items += len(batch)
        for job, artifact in zip(batch, artifacts):
            self._complete_batch_job(job, artifact)

    def _run_batch_job_one(self, job: _BatchJob[InputT, StoredT]) -> None:
        try:
            artifact = self._hook.encode_one(job.item)
        except BaseException as exc:
            self._fail_batch_job(job, exc, phase="encode")
            return
        self._complete_batch_job(job, artifact)

    def _complete_batch_job(
        self,
        job: _BatchJob[InputT, StoredT],
        artifact: ArtifactT,
    ) -> None:
        try:
            stored = self._hook.store_artifact(artifact)
            should_cache = self._hook.revalidate(job.item, job.key)
            with self._lock:
                if should_cache:
                    self._cache.put(job.cache_key, stored)
                self._inflight.pop(job.cache_key, None)
        except BaseException as exc:
            self._fail_batch_job(job, exc, phase="post_encode")
            return
        self._emit_event(
            job.request_id,
            "reference_encode_end",
            key=job.key,
            result="success",
            cacheable=True,
        )
        job.future.set_result(stored)

    def _fail_batch_job(
        self,
        job: _BatchJob[InputT, StoredT],
        exc: BaseException,
        *,
        phase: str,
    ) -> None:
        self._add_exception_note(exc, job.desc)
        with self._lock:
            self._inflight.pop(job.cache_key, None)
            self._failed += 1
        self._emit_event(
            job.request_id,
            "reference_encode_end",
            key=job.key,
            result="error",
            error_type=type(exc).__name__,
        )
        self._emit_event(
            job.request_id,
            "reference_encode_failure",
            key=job.key,
            result="error",
            phase=phase,
            error_type=type(exc).__name__,
        )
        job.future.set_exception(exc)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
                "evictions": self._cache.eviction_count,
                "failed": self._failed,
                "uncacheable": self._uncacheable,
                "batches": self._batches,
                "batched_items": self._batched_items,
                "batch_fallbacks": self._batch_fallbacks,
            }

    @staticmethod
    def _add_exception_note(exc: BaseException, desc: str | None) -> None:
        if not desc:
            return
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(f"Reference encode context: {desc}")

    def _maybe_log(self) -> None:
        if self._log_prefix is None:
            return
        now = time.monotonic()
        if now - self._last_log_time < self._LOG_INTERVAL_S:
            return
        with self._lock:
            if now - self._last_log_time < self._LOG_INTERVAL_S:
                return
            self._last_log_time = now
            stats = {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
                "evictions": self._cache.eviction_count,
                "failed": self._failed,
                "uncacheable": self._uncacheable,
                "batches": self._batches,
                "batched_items": self._batched_items,
                "batch_fallbacks": self._batch_fallbacks,
            }
        logger.info("%s reference encode stats: %s", self._log_prefix, stats)

    def close(self) -> None:
        if self._batch_queue is None or self._batch_thread is None:
            return
        self._batch_queue.put(self._BATCH_STOP)
        self._batch_thread.join(timeout=1.0)
