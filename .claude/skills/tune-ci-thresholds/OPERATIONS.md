# Calibration operations

## GPU groups and layouts

Pin every calibration process with `TUNE_GPU_INCLUDE`. Use a separate run
directory per process. Do not point two processes at one plan.

Layouts (see `SKILL.md` for Mode A / B / C):

```bash
# Mode A — one group
TUNE_GPU_INCLUDE=0,1 python tune.py --model omni run --stages ALL --repeats 5 ...

# Mode C — N independent full calibrations (preferred multi-GPU default)
TUNE_GPU_INCLUDE=0,1 python tune.py ... --output-dir "$RUN_G01"
TUNE_GPU_INCLUDE=2,3 python tune.py ... --output-dir "$RUN_G23"
TUNE_GPU_INCLUDE=4,5 python tune.py ... --output-dir "$RUN_G45"

# Mode B — shared scope, disjoint stages, then merge-runs
TUNE_GPU_INCLUDE=0,1 python tune.py ... --stages <A> --output-dir "$RUN_A"
TUNE_GPU_INCLUDE=2,3 python tune.py ... --stages <B> --output-dir "$RUN_B"
```

Do not hard-code a fixed “two groups share ASR/TTS/Omni” partition table in
agent plans. Choose Mode C unless the user explicitly wants one combined
worst-of-five via Mode B.

## Concurrent isolation (required for any multi-group run)

These rules apply whether groups run Mode B or Mode C.

1. **Disjoint GPUs.** Include sets must not overlap. Respect `TUNE_GPU_EXCLUDE`
   for host-reserved devices; never launch on or clean those GPUs.
2. **Per-group cache root.** Give each group a distinct `XDG_CACHE_HOME` and/or
   `HOME` (or equivalent `OMNI_CI_HOME` partition). FlashInfer cleanup wipes
   only this job’s first cache dir; wiping every candidate path races live
   workers on other groups.
3. **Scoped cleanup only.** Every cleanup path must pass physical GPU ids via
   `CUDA_VISIBLE_DEVICES`. See Cleanup below.
4. **No interactive shell pollution.** Bootstrap from
   `.github/scripts/ci_env.sh` (and explicit exports). Do **not** `source`
   `~/.zshrc` / `~/.bashrc` for calibration — they often force
   `CUDA_VISIBLE_DEVICES` and break multi-group pinning.
5. **Secrets.** Keep `HF_TOKEN` in a dedicated file with mode `600`, or the
   process environment. Do not rely on interactive shell state.
6. **Auditable cleanup.** Keep cleanup stdout/stderr visible so cross-group
   mis-kills are diagnosable.

## Cleanup

`tune.py` passes the actual physical GPU indices from the pytest launch to
`delete_gpu_process.sh --kill-orphans`. Invariants:

- `CUDA_VISIBLE_DEVICES` selects **physical** GPU indices to clean.
- The script then **unsets** CVD before calling `nvidia-smi`, so `--id=N` is
  never remapped by a visible-device subset.
- Unscoped cleanup is refused unless `GITHUB_ACTIONS=true`,
  `OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN=1`, or CVD is explicitly `all`.
- Orphan kill matches only `/dev/nvidiaN` for selected ids (not nvidiactl/uvm).
- Skip processes whose own CVD is **disjoint** from the cleanup scope.
- Skip ephemeral version probes (`importlib.metadata`, `import sglang`,
  `m.version(...)`) so one group’s cleanup cannot SIGKILL another group’s
  precheck.
- Use `/usr/bin/tr` inside the script; interactive shells may alias `tr`→`tree`.

`benchmarks/benchmarker/utils.wait_for_gpu_memory_release` requires
`CUDA_VISIBLE_DEVICES` when not on GitHub Actions.

Manual cleanup:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash .github/scripts/delete_gpu_process.sh --kill-orphans
```

Never use unscoped `pkill -9`, `killall`, or a cleanup command without an
explicit target set on a shared host.

## Monitoring

Create exactly one Tab A and one Tab B for every configured GPU group. Three
groups require three Tab A terminals and three Tab B terminals, even when a
group is temporarily idle between queued runs. Watcher count follows the number
of GPU groups, not the Mode A/B/C choice.

Tab A — aggregate strict progress:

```bash
bash .claude/skills/tune-ci-thresholds/watch_calibration_group.sh \
  <gpu-group> <group-run-1> [<group-run-2> ...]
```

Tab B — active server / pytest logs:

```bash
bash .claude/skills/tune-ci-thresholds/watch_calibration_servers.sh \
  <gpu-group> <group-run-1> [<group-run-2> ...]
```

Behavior:

- Resolves the active pytest from its process and `--basetemp`.
- Prefers `server.log` under that basetemp. Locally (non-CI),
  `server_log_file()` returns `None`, so router/worker stdout is multiplexed
  into the sibling pytest `runN.log`; Tab B falls back to that file.
- Detaches when cleanup kills a server or pytest exits; attaches the next
  launch in the same Tab B (must not stay stuck on a completed log).
- IDE terminals truncate around ~1MiB. Tab B filters Decode/Prefill batch spam
  by default and tees a durable copy to
  `/tmp/calibration_tabB_<gpu-group>.log` (example:
  `/tmp/calibration_tabB_0_1.log`). Set `CALIBRATION_SERVER_WATCH_VERBOSE=1`
  for the raw terminal stream.

Start both watchers before that group’s first job. Pass all run directories
assigned to the group, including queued dirs whose `plan.json` does not exist
yet. Keep watchers alive until the group’s queue is complete.

Also poll `status`, `strict-audit`, and `nvidia-smi` at least every 120 seconds
while work is active. The legacy `tail_calibration_pytest.sh` is a debugging
fallback only.

## CUDA recovery

If `nvidia-smi` works but PyTorch cannot initialize CUDA, stop the affected
group. Do not loop retries or broaden cleanup. Host recovery may require
restarting Fabric Manager and the container. Re-run the CUDA smoke and precheck
before resume.

## Split-run reporting (Mode B only)

Merge strict-ready stage partitions with:

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

Independent full calibrations (Mode C) remain separate replications. Comparing
their distributions is useful; silently combining them changes N and the
statistical policy.
