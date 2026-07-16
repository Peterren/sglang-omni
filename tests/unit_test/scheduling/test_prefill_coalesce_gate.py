# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the prefill admission-coalescing gate.

The gate holds prefill until ``prefill_coalesce_requests`` are waiting or the
oldest has waited ``prefill_coalesce_wait_ms``; chunked prefill in flight and
an empty queue pass straight through. Tested against a stub scheduler so no
engine is needed — the upstream call is patched to a sentinel.
"""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

_UPSTREAM_BATCH = object()


class _StubScheduler:
    """The attribute surface get_new_batch_prefill touches."""

    def __init__(self, *, coalesce_requests: int, wait_ms: float = 60.0) -> None:
        self.prefill_coalesce_requests = coalesce_requests
        self.prefill_coalesce_wait_s = wait_ms / 1e3
        self._prefill_coalesce_t0: float | None = None
        self.chunked_req = None
        self.waiting_queue: list[object] = []

    def get_new_batch_prefill(self):
        return OmniScheduler.get_new_batch_prefill(self)


@pytest.fixture()
def upstream():
    with mock.patch.object(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        return_value=_UPSTREAM_BATCH,
    ) as patched:
        yield patched


def test_disabled_gate_passes_through(upstream):
    sched = _StubScheduler(coalesce_requests=0)
    sched.waiting_queue = [object()]
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_chunked_prefill_bypasses_gate(upstream):
    sched = _StubScheduler(coalesce_requests=8)
    sched.waiting_queue = [object()]
    sched.chunked_req = object()
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_empty_queue_passes_through_and_resets_timer(upstream):
    sched = _StubScheduler(coalesce_requests=8)
    sched._prefill_coalesce_t0 = 123.0
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert sched._prefill_coalesce_t0 is None


def test_full_batch_passes_through_immediately(upstream):
    sched = _StubScheduler(coalesce_requests=4)
    sched.waiting_queue = [object()] * 4
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert sched._prefill_coalesce_t0 is None


def test_small_queue_is_held_until_deadline(upstream):
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    sched.waiting_queue = [object()] * 2
    with mock.patch.object(omni_scheduler.time, "perf_counter") as clock:
        clock.return_value = 100.0
        assert sched.get_new_batch_prefill() is None  # arms the timer
        assert sched._prefill_coalesce_t0 == 100.0

        clock.return_value = 100.03  # inside the 60ms window
        assert sched.get_new_batch_prefill() is None

        clock.return_value = 100.07  # deadline expired
        assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
        assert sched._prefill_coalesce_t0 is None
    upstream.assert_called_once()


def test_reaching_target_releases_before_deadline(upstream):
    sched = _StubScheduler(coalesce_requests=3, wait_ms=60.0)
    sched.waiting_queue = [object()]
    with mock.patch.object(omni_scheduler.time, "perf_counter") as clock:
        clock.return_value = 5.0
        assert sched.get_new_batch_prefill() is None

        sched.waiting_queue = [object()] * 3  # target reached, clock unchanged
        assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
        assert sched._prefill_coalesce_t0 is None
