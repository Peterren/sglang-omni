# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, TypeVar

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
StoredT = TypeVar("StoredT")

logger = logging.getLogger(__name__)


def _identity(value: Any) -> Any:
    return value


def _clone_tensor_values(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tensor_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_tensor_values(item) for item in value)
    return value


class BatchedReferenceEncoder(Generic[InputT, OutputT]):
    """Coalesce concurrent reference encodes into one batched worker call."""

    def __init__(
        self,
        batch_encode_fn: Callable[[list[InputT]], Sequence[OutputT]],
        *,
        single_encode_fn: Callable[[InputT], OutputT] | None = None,
        validate_fn: Callable[[InputT], None] | None = None,
        key_fn: Callable[[InputT], Hashable] | None = None,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
        encode_timeout_s: float = 120.0,
        worker_name: str = "reference-encode",
    ) -> None:
        self._batch_encode_fn = batch_encode_fn
        self._single_encode_fn = single_encode_fn
        self._validate_fn = validate_fn
        self._key_fn = key_fn or (lambda item: item)  # type: ignore[return-value]
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._encode_timeout_s = float(encode_timeout_s)
        self._queue: queue.Queue[tuple[InputT, concurrent.futures.Future[OutputT]]] = (
            queue.Queue()
        )
        self._thread = threading.Thread(
            target=self._worker, name=worker_name, daemon=True
        )
        self._thread.start()

    def encode(self, item: InputT) -> OutputT:
        if self._validate_fn is not None:
            self._validate_fn(item)
        future: concurrent.futures.Future[OutputT] = concurrent.futures.Future()
        self._queue.put((item, future))
        return future.result(timeout=self._encode_timeout_s)

    def _drain_batch(
        self,
    ) -> list[tuple[InputT, concurrent.futures.Future[OutputT]]]:
        batch = [self._queue.get()]
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    batch.append(self._queue.get(timeout=self._max_wait_s))
                else:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            batch = self._drain_batch()
            unique_items: list[InputT] = []
            unique_keys: list[Hashable] = []
            seen: set[Hashable] = set()
            for item, _ in batch:
                key = self._key_fn(item)
                if key in seen:
                    continue
                seen.add(key)
                unique_items.append(item)
                unique_keys.append(key)

            results: dict[Hashable, OutputT | BaseException] = {}
            try:
                encoded = list(self._batch_encode_fn(unique_items))
                if len(encoded) != len(unique_items):
                    raise RuntimeError(
                        "batched reference encode returned "
                        f"{len(encoded)} results for {len(unique_items)} inputs"
                    )
                results = dict(zip(unique_keys, encoded))
            except Exception:
                logger.exception("Batched reference encode failed; retrying per item")
                for key, item in zip(unique_keys, unique_items):
                    try:
                        results[key] = self._encode_single(item)
                    except Exception as exc:
                        results[key] = exc

            for item, future in batch:
                key = self._key_fn(item)
                outcome = results.get(key)
                if isinstance(outcome, BaseException):
                    future.set_exception(
                        RuntimeError(f"reference encode failed for {item}: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(
                        RuntimeError(f"reference encode produced no result: {item}")
                    )
                else:
                    future.set_result(outcome)

    def _encode_single(self, item: InputT) -> OutputT:
        if self._single_encode_fn is not None:
            return self._single_encode_fn(item)
        encoded = list(self._batch_encode_fn([item]))
        if len(encoded) != 1:
            raise RuntimeError(
                "single reference encode returned "
                f"{len(encoded)} results for 1 input"
            )
        return encoded[0]


class ReferenceEncodeCache(Generic[StoredT]):
    """Content-addressed LRU cache with single-flight miss coalescing."""

    def __init__(
        self,
        *,
        max_items: int | None = 256,
        max_bytes: int | None = 64 * 1024 * 1024,
        cache_device: torch.device | str | None = "cpu",
        size_fn: Callable[[Any], int] | None = None,
        store_fn: Callable[[Any], StoredT] | None = None,
        load_fn: Callable[[StoredT], Any] | None = None,
        timeout_s: float = 130.0,
        log_interval_s: float | None = None,
        log_prefix: str | None = None,
    ) -> None:
        if max_items is not None and max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        self._cache = StageOutputCache(
            max_size=max_items,
            max_bytes=max_bytes,
            cache_device=cache_device,
            size_fn=size_fn,
        )
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future[StoredT]] = {}
        self._store_fn = store_fn or _identity
        self._load_fn = load_fn or _clone_tensor_values
        self._timeout_s = float(timeout_s)
        self._log_interval_s = log_interval_s
        self._log_prefix = log_prefix
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._last_log_time = 0.0

    @property
    def cache(self) -> StageOutputCache:
        return self._cache

    @property
    def inflight(self) -> dict[str, concurrent.futures.Future[StoredT]]:
        return self._inflight

    def get_or_encode(
        self,
        key: str | None,
        encode_fn: Callable[[], Any],
        *,
        desc: str,
        revalidate: Callable[[], bool] | None = None,
    ) -> Any:
        if key is None:
            return self._load_fn(self._store_fn(encode_fn()))

        key = str(key)
        stored: StoredT | None = None
        leader_fut: concurrent.futures.Future[StoredT] | None = None
        follower_fut: concurrent.futures.Future[StoredT] | None = None

        with self._lock:
            stored = self._cache.get(key)
            if stored is not None:
                self._hits += 1
            elif key in self._inflight:
                self._merged += 1
                follower_fut = self._inflight[key]
            else:
                self._misses += 1
                leader_fut = concurrent.futures.Future()
                self._inflight[key] = leader_fut

        if stored is not None:
            self._maybe_log()
            return self._load_fn(stored)

        if follower_fut is not None:
            try:
                stored = follower_fut.result(timeout=self._timeout_s)
            except Exception as cause:
                raise RuntimeError(
                    f"reference encode failed for {desc}: {cause}"
                ) from cause
            return self._load_fn(stored)

        assert leader_fut is not None
        try:
            result = encode_fn()
            stored = self._store_fn(result)
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
            leader_fut.set_exception(exc)
            raise

        do_put = revalidate() if revalidate is not None else True
        with self._lock:
            if do_put:
                self._cache.put(key, stored)
            self._inflight.pop(key, None)
        leader_fut.set_result(stored)
        self._maybe_log()
        return self._load_fn(stored)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
            }

    def _maybe_log(self) -> None:
        if self._log_interval_s is None or self._log_prefix is None:
            return
        now = time.monotonic()
        if now - self._last_log_time < self._log_interval_s:
            return
        with self._lock:
            if now - self._last_log_time < self._log_interval_s:
                return
            self._last_log_time = now
            snapshot = {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
            }
        logger.info(
            "%s: hits=%d misses=%d merged=%d entries=%d bytes=%d",
            self._log_prefix,
            snapshot["hits"],
            snapshot["misses"],
            snapshot["merged"],
            snapshot["entries"],
            snapshot["bytes"],
        )
