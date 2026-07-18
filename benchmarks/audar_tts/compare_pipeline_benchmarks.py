# SPDX-License-Identifier: Apache-2.0
"""Run and summarize the paired Audar-TTS refactor benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

RUN_ORDER = (
    ("latest", "latest-order1"),
    ("pre_t1", "pre-t1-order1"),
    ("pre_t1", "pre-t1-order2"),
    ("latest", "latest-order2"),
)
METRICS = (
    "total_s",
    "rtf",
    "engine_wall_s",
    "engine_reported_s",
    "engine_tokens_per_s",
    "reference_s",
    "vocoder_s",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-t1-checkout", type=Path, required=True)
    parser.add_argument("--latest-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def _commit(checkout: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_snapshot(
    *,
    checkout: Path,
    label: str,
    output_dir: Path,
    python: str,
    repeats: int,
) -> dict[str, Any]:
    runner = checkout / "benchmarks/audar_tts/run_pipeline_benchmark.py"
    if not runner.is_file():
        raise FileNotFoundError(f"benchmark runner not found: {runner}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(checkout)
    completed = subprocess.run(
        [
            python,
            str(runner),
            "--label",
            label,
            "--output-dir",
            str(output_dir),
            "--repeats",
            str(repeats),
        ],
        cwd=checkout,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    (output_dir / f"{label}.stdout").write_text(completed.stdout, encoding="utf-8")
    return json.loads((output_dir / f"{label}.json").read_text(encoding="utf-8"))


def _warm_medians(result: dict[str, Any]) -> dict[str, float]:
    iterations = result["iterations"]
    warm = iterations[1:] if len(iterations) > 1 else iterations
    return {
        metric: statistics.median(float(item[metric]) for item in warm)
        for metric in METRICS
    }


def _mean_metrics(runs: list[dict[str, float]]) -> dict[str, float]:
    return {metric: statistics.fmean(run[metric] for run in runs) for metric in METRICS}


def _percent_delta(latest: float, baseline: float) -> float:
    return (latest / baseline - 1.0) * 100.0


def main() -> None:
    args = _parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to exclude warmup")
    checkouts = {
        "pre_t1": args.pre_t1_checkout.resolve(),
        "latest": args.latest_checkout.resolve(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_runs: dict[str, dict[str, Any]] = {}
    medians: dict[str, list[dict[str, float]]] = {"pre_t1": [], "latest": []}
    for snapshot, label in RUN_ORDER:
        result = _run_snapshot(
            checkout=checkouts[snapshot],
            label=label,
            output_dir=output_dir,
            python=args.python,
            repeats=args.repeats,
        )
        raw_runs[label] = result
        medians[snapshot].append(_warm_medians(result))

    aggregate = {
        snapshot: _mean_metrics(snapshot_runs)
        for snapshot, snapshot_runs in medians.items()
    }
    hash_pairs = {
        (iteration["audio_code_sha256"], iteration["waveform_sha256"])
        for result in raw_runs.values()
        for iteration in result["iterations"]
    }
    summary = {
        "schema_version": 1,
        "run_order": [label for _, label in RUN_ORDER],
        "repeats_per_run": args.repeats,
        "warmup_iterations_excluded_per_run": 1,
        "warm_requests_per_snapshot": 2 * (args.repeats - 1),
        "commits": {
            snapshot: _commit(checkout) for snapshot, checkout in checkouts.items()
        },
        "pre_t1": {
            **aggregate["pre_t1"],
            "run_medians": medians["pre_t1"],
        },
        "latest": {
            **aggregate["latest"],
            "run_medians": medians["latest"],
        },
        "latest_vs_pre_percent": {
            metric: _percent_delta(
                aggregate["latest"][metric], aggregate["pre_t1"][metric]
            )
            for metric in METRICS
        },
        "all_hash_pairs_identical": len(hash_pairs) == 1,
        "hash_pairs": [
            {"audio_code_sha256": code_hash, "waveform_sha256": waveform_hash}
            for code_hash, waveform_hash in sorted(hash_pairs)
        ],
    }
    output_path = output_dir / "performance_summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
