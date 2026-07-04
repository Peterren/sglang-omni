# H100 CI Runner Capacity

This note documents how Omni CI uses the H100 self-hosted GitHub Actions
runner and what must be true before adding more runner capacity.

## Current Runner Lane

Omni GPU CI jobs use the GitHub Actions label set `self-hosted,h100`.
The current H100 lane is served by the `omni-runner-h100` worker and the
workflow containers pin themselves to GPU devices `6,7` through
`NVIDIA_VISIBLE_DEVICES=6,7`.

That means queue time is controlled by GitHub Actions runner-worker
availability, not by raw GPU idleness. A host can show idle H100 GPUs while
Actions still keeps jobs queued if no online worker with the required labels is
available.

The current pinning is intentional: all Omni CI stages assume the same
two-GPU slice and the shared `/github/home` plus Hugging Face cache layout. Do
not start more `self-hosted,h100` workers on the same host unless each worker
has an explicit non-overlapping GPU slice and the workflows route jobs to that
slice.

## Queue Diagnosis

Use the run jobs API to distinguish runner queue time from benchmark runtime:

```bash
gh api /repos/sgl-project/sglang-omni/actions/runs/<run-id>/jobs \
  --jq '.jobs[] | {id,name,status,conclusion,labels,runner_name,started_at,completed_at}'
```

Interpretation:

- `status=queued`, empty `runner_name`: GitHub Actions has accepted the job but
  has not assigned a matching runner worker.
- `status=in_progress`, non-empty `runner_name`: the job has an Actions worker;
  inspect the runner tmux or the job log for runtime issues.
- Long queue time followed by a short job duration is a runner-capacity problem,
  not a benchmark regression.

For a representative case, the baseline MOSS run in
[#943](https://github.com/sgl-project/sglang-omni/issues/943) waited about 49
minutes for `setup - omni venv` to start on `omni-runner-h100`, then completed
the setup in about 1 minute. Its TTS streaming and non-streaming benchmark
stages each took about 5 minutes once assigned.

## Safe Scaling Options

Adding capacity requires both runner registration and GPU placement discipline.

1. Add another H100 Actions worker on a separate host that can safely expose the
   same GPU devices `6,7` to Omni CI containers. This preserves the current
   workflow pinning and is the lowest-risk way to increase `self-hosted,h100`
   concurrency.
2. Split one host into multiple runner workers only after adding slice-specific
   routing. Example labels are `h100-gpu01`, `h100-gpu23`, `h100-gpu45`, and
   `h100-gpu67`. Each runner must have a separate actions-runner directory,
   unique runner name, and a container GPU binding that cannot overlap another
   worker's binding.
3. If short TTS gates should not wait behind longer Qwen3-Omni stages, add
   job-class labels such as `h100-tts` and `h100-qwen3` and update the relevant
   reusable workflows to target those labels deliberately.

Avoid registering extra workers with only the generic `h100` label on the same
host while the workflows still hard-code `NVIDIA_VISIBLE_DEVICES=6,7`; that
can schedule concurrent jobs onto the same GPUs.

## Runner Maintenance Checklist

Before marking runner capacity as healthy, verify:

- The intended runner workers are online in GitHub Actions and have the expected
  labels.
- Each worker has a distinct runner name and actions-runner directory.
- Each worker's container GPU slice is explicit and non-overlapping.
- The runner can mount `/dev/shm`, `/github/home`, and the shared Hugging Face
  cache with the same paths expected by Omni CI.
- A manual MOSS Omni CI dispatch can reach `setup - omni venv` without waiting
  tens of minutes when the assigned GPU slice is idle.

If the project intentionally keeps a single H100 worker slot, treat long
`self-hosted,h100` queues as expected capacity limits and say so in PR reviews
instead of interpreting them as model or test failures.
