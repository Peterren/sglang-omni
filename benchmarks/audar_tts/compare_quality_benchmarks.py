# SPDX-License-Identifier: Apache-2.0
"""Generate the paired Audar Arabic quality corpus from two checkouts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-t1-checkout", type=Path, required=True)
    parser.add_argument("--latest-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--samples", type=int, default=50)
    return parser.parse_args()


def _run(
    *,
    runner: Path,
    checkout: Path,
    output_dir: Path,
    label: str,
    python: str,
    samples: int,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(checkout)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            python,
            str(runner),
            "--label",
            label,
            "--checkout",
            str(checkout),
            "--output-dir",
            str(output_dir),
            "--samples",
            str(samples),
        ],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    log = completed.stdout + completed.stderr
    (output_dir / "runner.log").write_text(log, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{label} quality generation failed:\n{log[-4000:]}")


def main() -> None:
    args = _parse_args()
    runner = Path(__file__).with_name("run_quality_benchmark.py").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        runner=runner,
        checkout=args.pre_t1_checkout.resolve(),
        output_dir=output_dir / "pre-t1",
        label="pre-t1",
        python=args.python,
        samples=args.samples,
    )
    _run(
        runner=runner,
        checkout=args.latest_checkout.resolve(),
        output_dir=output_dir / "latest",
        label="latest",
        python=args.python,
        samples=args.samples,
    )


if __name__ == "__main__":
    main()
