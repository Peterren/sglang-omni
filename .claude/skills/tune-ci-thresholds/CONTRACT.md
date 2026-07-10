# Calibration contract

These invariants define a valid calibration. CLI commands must enforce them;
agent instructions are not a substitute for code gates.

## Session identity

- A new request uses a new UTC-timestamped directory on current `HEAD`.
- Resume is interruption recovery for the same directory and commit.
- Every run artifact records the calibration commit.
- A calibration never mixes artifacts from commits, schemas, or environments.

## Observation validity

A stage repeat is strict-valid only when:

- its result JSON exists and is readable;
- every tracked metric is non-null;
- sample `ok == total`;
- `total == expected_samples` when configured;
- the recorded commit matches the plan.

Threshold assertion failures may still yield a valid observation when all
metrics and samples were produced. Infrastructure crashes, OOMs, timeouts,
missing output, and partial samples are not valid metric observations.

## Worst-of-N

- Default N is 5.
- Every selected stage needs exactly N strict-valid observations.
- Lower-bound metrics use the minimum; upper-bound metrics use the maximum.
- No partial or failed observation participates in aggregation.
- Outliers are flagged and retained unless a separately documented invalidation
  proves the run was not a valid observation.

## Schema

- `models/<model>/config.yaml` declares non-inferable metric paths and sample
  scopes.
- `stages.yaml` is generated deterministically from config and current tests.
- `CONCURRENCY` is execution fan-out, never sample count.
- A test or threshold-file hash mismatch blocks calibration.
- Report and apply consume the schema bound to the run.

## GPU ownership

- A pytest invocation owns only its selected physical GPU indices.
- Cleanup may target only those indices.
- Concurrent groups require disjoint `TUNE_GPU_INCLUDE` values.
- Global process-pattern and user-wide kills are forbidden.

## Final consumers

`report` and `apply-plan` use the same readiness validator. Neither may consume
an incomplete run. Apply writes raw pre-slack references and must never write a
derived assertion threshold.

## Required provenance

The final artifact records commit, dirty state, venv, dependency hash, core
versions, container identity when available, driver/GPU/topology, selected GPU
group, relevant environment, required model/dataset IDs, attempt history, and
seed policy.
