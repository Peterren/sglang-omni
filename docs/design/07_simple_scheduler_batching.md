# 07_simple_scheduler_batching

## SimpleScheduler Concurrent Batching

Status: Draft

Top-level module: M7

Behavior level: Red

Issue: TBD

PR: TBD

## Goal

Support `max_concurrency > 1` together with `batch_compute_fn` when there is a
real consumer and profiling evidence.

## Scope

- Queue-agnostic batch collection for `SimpleScheduler`.
- Parity between the serial inbox path and the concurrent `async_inbox` path.
- Ordering, cancellation, and error semantics for concurrent batches.
- `max_batch_wait_ms`, `request_cost_fn`, and `max_batch_cost` behavior under
  concurrent collection.

## Non-scope

- This is not a simple deletion of the current guard.
- Do not mix this with T2 or T5 work.
- Do not implement runtime scheduler changes without a consumer or profiling
  evidence.

## Contract

The scheduler owns batch collection semantics. Call sites only declare
`batch_compute_fn` and optional cost functions. They should not need stage-local
queue logic to make concurrent batching correct.

## Current State

`SimpleScheduler` already supports serial local batching with `batch_compute_fn`
when `max_concurrency=1`. It also supports concurrent non-batched execution with
`max_concurrency > 1`.

The combination is intentionally rejected today because the concurrent path
bridges the blocking inbox into an internal `async_inbox` and then lets worker
tasks claim individual messages. Removing the guard would make multiple workers
race to form batches, which can break exclusive claim, FIFO expectations, cost
accounting, and per-request error ownership.

## Migration Plan

1. Attach consumer or profile evidence. M4B is one possible consumer, but only
   after reference-encode profiling shows different-key batching is useful.
2. Write a design for the queue-agnostic collector and get it reviewed.
3. Add contract tests first in an isolated red PR.
4. Modify `SimpleScheduler` only after the contract is accepted.
5. Keep the old `max_concurrency=1` batching behavior unchanged.

## Candidate Design

Use one collector owner per scheduler instance, not one collector per worker.
The collector should read from the same source of truth for both serial and
concurrent modes, form batches according to size, wait, and cost limits, and
hand complete work items to workers.

The collector should produce either a single-message work item or a batch work
item. Workers run `compute_fn` for single items and `batch_compute_fn` for batch
items. This preserves the call-site contract and keeps batching decisions inside
the scheduler.

Abort and error handling stay per request. A failed batch emits one error per
claimed request. An aborted request is consumed before batch execution when
possible, and after execution before result emission when the abort races with
compute.

## Contract Tests

- Exclusive claim: each request is assigned to exactly one work item.
- FIFO: batch formation preserves inbox order for eligible `new_request`
  messages.
- Exception propagation: a batch failure emits per-request errors for all
  claimed requests.
- Cancellation: aborted requests are not emitted as successes.
- No busy loop: idle collectors and workers block or time out normally.
- Wait parity: `max_batch_wait_ms` behaves the same in serial and concurrent
  modes.
- Cost parity: `request_cost_fn` and `max_batch_cost` split batches the same way
  in serial and concurrent modes.
- Non-request messages: non-`new_request` messages do not corrupt a pending
  batch.

## Open Decisions

- Owner TBD: confirm whether this module already has a GitHub issue.
- Owner TBD: migrate unresolved comments from the old RFC into this document.
- Owner TBD: confirm the behavior class before opening the red PR.

## Done

- [ ] Consumer or profile evidence attached.
- [ ] Queue-agnostic collector design approved.
- [ ] Red PR isolated.
- [ ] All concurrency invariants tested.
