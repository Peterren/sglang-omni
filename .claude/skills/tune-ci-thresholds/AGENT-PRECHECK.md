# Calibration precheck

Run this checklist before every fresh session and after environment recovery.
`tune.py run` must not start while a mandatory gate fails.

## 1. Scope and provenance

- Confirm model, selected stages, and repeats. Default repeats: 5.
- Record `git rev-parse HEAD`.
- Use a fresh `.tune-runs/<UTC>_<label>/` unless explicitly resuming.
- Resume only when `HEAD` matches the run plan.
- Regenerate `stages.yaml` after relevant test/config changes.

## 2. GPU ownership

- Set `TUNE_GPU_INCLUDE` to the exact group owned by this calibration process,
  normally two GPUs such as `0,1`.
- Set `TUNE_GPU_EXCLUDE` for host-reserved GPUs, normally `6,7`.
- Concurrent processes must use disjoint include sets and run directories.
- Verify every selected GPU is idle and below 2048 MiB before launch.
- Never free GPUs with global `pkill` or user-wide process kills.

```bash
export TUNE_GPU_INCLUDE=0,1
export TUNE_GPU_EXCLUDE=6,7
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

## 3. CUDA smoke

`nvidia-smi` is insufficient. Initialize CUDA through the calibration venv:

```bash
"$TUNE_VENV_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
print(torch.__version__, torch.cuda.device_count())
PY
```

On cu130 environments, ensure the venv CUDA libraries are on
`LD_LIBRARY_PATH` when required by the host image.

## 4. Repo and dependencies

- Repo contains `pyproject.toml` at the selected commit.
- Calibration venv exists.
- `torch` and `sglang` match current project pins.
- Editable package points to the selected worktree.
- `CAP_SYS_PTRACE` is present for the FP8 TP=2 test.

Using the maintained calibration venv normally requires only:

```bash
cd "$TUNE_REPO_ROOT"
uv pip install -e .
```

Do not rebuild the venv or bulk-download assets before precheck identifies a
specific gap.

## 5. Caches and assets

- Required Hugging Face model and dataset snapshots are locally available.
- Speaker-similarity weights and completion marker exist for TTS stages.
- UTMOS assets are warmed before TTS calibration.
- `OMNI_CI_HOME/.cache` and `.torchinductor` are writable.

## 6. Official precheck

Run it for each selected model:

```bash
python .claude/skills/tune-ci-thresholds/tune.py \
  --model <model> precheck --output-dir "$RUN"
```

Pass criteria:

- precheck exits zero;
- core dependency pins match;
- enough GPUs exist inside `TUNE_GPU_INCLUDE`;
- required models/datasets and metric assets are present;
- `environment-fingerprint.json` is written;
- any unverified image identity is explicitly visible.

## 7. Active supervision

During a run, poll at most every 120 seconds:

```bash
python .claude/skills/tune-ci-thresholds/tune.py status --run-dir "$RUN"
python .claude/skills/tune-ci-thresholds/tune.py strict-audit --run-dir "$RUN"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
bash .claude/skills/tune-ci-thresholds/tail_calibration_pytest.sh "$RUN"
```

Stop on CUDA initialization failure, extraction warnings, wrong sample scope,
or cleanup affecting GPUs outside the configured group.

## 8. Completion

Before report or apply:

- every selected stage has N/N strict observations;
- every observation has full expected sample scope and all metrics;
- git provenance passes;
- `report` succeeds through `validate_run_ready()`;
- no calibration or pytest process remains alive.
