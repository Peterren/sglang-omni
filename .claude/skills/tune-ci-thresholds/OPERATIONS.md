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

Use `status`, `strict-audit`, the active pytest log, and `nvidia-smi`. Treat the
milestone log and verbose pytest log as separate streams. Poll at least every
120 seconds while work is active.

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
