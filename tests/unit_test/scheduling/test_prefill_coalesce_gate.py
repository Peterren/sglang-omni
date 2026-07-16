# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the prefill admission-coalescing gate.

The gate holds prefill until ``prefill_coalesce_requests`` are waiting or the
oldest queued request has waited ``prefill_coalesce_wait_ms``. The deadline is
keyed on each request's enqueue time (``_coalesce_enqueue_t``), so partial
upstream admission or an aborted request never restarts the window for the
requests left behind. Chunked prefill in flight and an empty queue pass
straight through. Tested against a stub scheduler with the upstream call
patched to a sentinel.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

_UPSTREAM_BATCH = object()


def _req(enqueue_t: float | None):
    if enqueue_t is None:
        return SimpleNamespace()
    return SimpleNamespace(_coalesce_enqueue_t=enqueue_t)


class _StubScheduler:
    """The attribute surface get_new_batch_prefill touches."""

    def __init__(self, *, coalesce_requests: int, wait_ms: float = 60.0) -> None:
        self.prefill_coalesce_requests = coalesce_requests
        self.prefill_coalesce_wait_s = wait_ms / 1e3
        self.chunked_req = None
        self.waiting_queue: list = []

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


@pytest.fixture()
def clock():
    with mock.patch.object(omni_scheduler.time, "perf_counter") as patched:
        patched.return_value = 100.0
        yield patched


def test_disabled_gate_passes_through(upstream):
    sched = _StubScheduler(coalesce_requests=0)
    sched.waiting_queue = [_req(0.0)]
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_chunked_prefill_bypasses_gate(upstream):
    sched = _StubScheduler(coalesce_requests=8)
    sched.waiting_queue = [_req(0.0)]
    sched.chunked_req = object()
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_empty_queue_passes_through(upstream):
    sched = _StubScheduler(coalesce_requests=8)
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_full_batch_passes_through_immediately(upstream):
    sched = _StubScheduler(coalesce_requests=4)
    sched.waiting_queue = [_req(100.0)] * 4
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_small_queue_is_held_until_oldest_expires(upstream, clock):
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    sched.waiting_queue = [_req(100.0), _req(100.01)]

    clock.return_value = 100.03  # oldest has waited 30ms of the 60ms window
    assert sched.get_new_batch_prefill() is None

    clock.return_value = 100.07  # oldest past the deadline
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    upstream.assert_called_once()


def test_reaching_target_releases_before_deadline(upstream, clock):
    sched = _StubScheduler(coalesce_requests=3, wait_ms=60.0)
    sched.waiting_queue = [_req(100.0)]
    assert sched.get_new_batch_prefill() is None

    sched.waiting_queue = [_req(100.0)] * 3  # target reached, clock unchanged
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_partial_admission_leftovers_keep_their_deadline(upstream, clock):
    # Upstream admitted part of an expired wave; the leftovers still carry old
    # enqueue times and must NOT re-wait a fresh window.
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.1
    sched.waiting_queue = [_req(100.0), _req(100.02)]  # both past the deadline
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_abort_does_not_hand_newcomers_an_expired_deadline(upstream, clock):
    # The request that opened the window was aborted; a fresh arrival must
    # wait its own window rather than inherit the nearly expired one.
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.1
    sched.waiting_queue = [_req(100.09)]  # newcomer, 10ms old
    assert sched.get_new_batch_prefill() is None


def test_unstamped_request_is_treated_as_fresh(upstream, clock):
    sched = _StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 200.0
    sched.waiting_queue = [_req(None)]
    assert sched.get_new_batch_prefill() is None
