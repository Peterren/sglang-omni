# Calibration operations

## GPU groups

Pin every calibration process with `TUNE_GPU_INCLUDE`.

One group:

```bash
TUNE_GPU_INCLUDE=0,1 python tune.py --model omni run --stages ALL --repeats 5 ...
```

Split stages across two groups:

```bash
TUNE_GPU_INCLUDE=0,1 python tune.py --model omni run --stages <A> --repeats 5 ...
TUNE_GPU_INCLUDE=2,3 python tune.py --model omni run --stages <B> --repeats 5 ...
```

Independent replications:

```bash
TUNE_GPU_INCLUDE=0,1 python tune.py --model omni run --stages ALL --repeats 5 ...
TUNE_GPU_INCLUDE=2,3 python tune.py --model omni run --stages ALL --repeats 5 ...
```

Use separate run directories. Do not run two processes against one plan.

## Cleanup

`tune.py` passes the actual physical GPU indices from the pytest launch to
`delete_gpu_process.sh --kill-orphans`. The script uses `CUDA_VISIBLE_DEVICES`
as its target set. It must not inspect or kill unrelated GPU groups.

If manual cleanup is required:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash .github/scripts/delete_gpu_process.sh --kill-orphans
```

Never use unscoped `pkill -9`, `killall`, or a cleanup command without an
explicit target set on a shared host.

## Monitoring

Create exactly one Tab A and one Tab B for every configured GPU group. Three GPU
groups require three Tab A terminals and three Tab B terminals, even when a
group is temporarily idle between queued runs.

Tab A shows aggregate strict progress for every run assigned to that GPU group:

```bash
bash .claude/skills/tune-ci-thresholds/watch_calibration_group.sh \
  <gpu-group> <group-run-1> [<group-run-2> ...]
```

Tab B follows server logs for the active pytest in that GPU group:

```bash
bash .claude/skills/tune-ci-thresholds/watch_calibration_servers.sh \
  <gpu-group> <group-run-1> [<group-run-2> ...]
```

The Tab B watcher resolves the active pytest from its process and `--basetemp`,
discovers every new `server.log`, and attaches it dynamically. When cleanup
kills a server or pytest exits, it detaches stale logs. A later server launch is
discovered and attached in the same Tab B; the terminal must never remain stuck
on a completed server log.

Start both watchers before launching that group's first calibration job. Pass
all run directories already assigned to the group, including queued directories
whose `plan.json` does not exist yet. Keep the watchers alive until the group's
entire queue is complete.

Also poll `status`, `strict-audit`, and `nvidia-smi` at least every 120 seconds
while work is active. The legacy `tail_calibration_pytest.sh` follows combined
pytest output and is only a debugging fallback; it does not replace either
group-level watcher.

## CUDA recovery

If `nvidia-smi` works but PyTorch cannot initialize CUDA, stop the affected
group. Do not loop retries or broaden cleanup. Host recovery may require
restarting Fabric Manager and the container. Re-run the CUDA smoke and precheck
before resume.

## Split-run reporting

Merge two strict-ready stage partitions with:

```bash
python tune.py merge-runs --run-dir "$RUN_A" --run-dir "$RUN_B" \
  --output-dir "$RUN_COMBINED"
```

The command validates:

- identical commit and schema hashes;
- compatible environment fingerprints;
- identical repeat policy;
- disjoint and complete stage ownership;
- strict readiness of both inputs.

Independent full calibrations remain separate replications. Comparing their
distributions is useful; silently combining them changes N and the statistical
policy.
