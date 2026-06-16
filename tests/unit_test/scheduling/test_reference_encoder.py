from __future__ import annotations

import threading
import time

import pytest
import torch

from sglang_omni.scheduling.reference_encoder import (
    BatchedReferenceEncoder,
    ReferenceEncodeCache,
)


def test_batched_reference_encoder_falls_back_and_isolates_failures() -> None:
    calls: list[list[str]] = []

    def batch_encode(items: list[str]) -> list[torch.Tensor]:
        calls.append(list(items))
        if len(items) > 1 and "bad" in items:
            raise RuntimeError("batch failed")
        out = []
        for item in items:
            if item == "bad":
                raise RuntimeError("cannot encode")
            out.append(torch.full((2, 3), len(item), dtype=torch.long))
        return out

    encoder = BatchedReferenceEncoder(
        batch_encode,
        max_batch_size=4,
        max_batch_wait_ms=20,
        encode_timeout_s=5.0,
        worker_name="test-reference-encode",
    )

    results: dict[str, object] = {}

    def run(item: str) -> None:
        try:
            results[item] = encoder.encode(item)
        except Exception as exc:
            results[item] = exc

    threads = [
        threading.Thread(target=run, args=(item,)) for item in ("aa", "bbb", "bad")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert torch.equal(results["aa"], torch.full((2, 3), 2, dtype=torch.long))
    assert torch.equal(results["bbb"], torch.full((2, 3), 3, dtype=torch.long))
    assert isinstance(results["bad"], RuntimeError)
    assert any(len(call) > 1 for call in calls)


def test_reference_encode_cache_single_flight_and_clone_policy() -> None:
    gate = threading.Event()
    call_count = 0

    cache = ReferenceEncodeCache(
        max_items=16,
        max_bytes=1 << 20,
        store_fn=lambda tensor: tensor.detach().to("cpu", dtype=torch.int32),
        load_fn=lambda tensor: tensor.clone().to(torch.long),
        timeout_s=5.0,
    )

    def encode() -> torch.Tensor:
        nonlocal call_count
        gate.wait()
        call_count += 1
        return torch.full((4, 3), 11, dtype=torch.long)

    results: list[torch.Tensor | None] = [None] * 6
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            results[index] = cache.get_or_encode("same-key", encode, desc="same-key")
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(idx,)) for idx in range(len(results))
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    gate.set()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert call_count == 1
    assert all(result is not None for result in results)
    assert len({result.data_ptr() for result in results if result is not None}) == len(
        results
    )

    first_hit = cache.get_or_encode("same-key", encode, desc="same-key")
    first_hit.fill_(-1)
    second_hit = cache.get_or_encode("same-key", encode, desc="same-key")

    assert torch.equal(second_hit, torch.full((4, 3), 11, dtype=torch.long))
    assert cache.stats() == {
        "hits": 2,
        "misses": 1,
        "merged": len(results) - 1,
        "entries": 1,
        "bytes": 4 * 3 * 4,
    }


def test_reference_encode_cache_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="max_items"):
        ReferenceEncodeCache(max_items=0, max_bytes=1024)
    with pytest.raises(ValueError, match="max_bytes"):
        ReferenceEncodeCache(max_items=1, max_bytes=0)


def test_reference_encode_cache_revalidate_failure_clears_inflight() -> None:
    cache = ReferenceEncodeCache(timeout_s=0.05)

    def encode() -> torch.Tensor:
        return torch.ones((1,), dtype=torch.long)

    def fail_revalidate() -> bool:
        raise RuntimeError("stat failed")

    with pytest.raises(RuntimeError, match="stat failed"):
        cache.get_or_encode(
            "flaky-key",
            encode,
            desc="flaky-key",
            revalidate=fail_revalidate,
        )

    assert cache.inflight == {}
    assert cache.stats()["entries"] == 0

    recovered = cache.get_or_encode(
        "flaky-key",
        encode,
        desc="flaky-key",
        revalidate=lambda: True,
    )
    assert torch.equal(recovered, torch.ones((1,), dtype=torch.long))
