# SPDX-License-Identifier: Apache-2.0
"""Summarize Audar Arabic and translated-English text metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sacrebleu.metrics import BLEU


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-t1-wer", type=Path, required=True)
    parser.add_argument("--pre-t1-translated-wer", type=Path, required=True)
    parser.add_argument("--latest-wer", type=Path, required=True)
    parser.add_argument("--latest-translated-wer", type=Path, required=True)
    parser.add_argument("--wav-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize(wer_path: Path, translated_path: Path) -> dict[str, Any]:
    wer = _load(wer_path)
    translated = _load(translated_path)
    if wer["summary"]["evaluated"] != 1 or translated["summary"]["evaluated"] != 1:
        raise ValueError(
            "quality regression artifact must contain one evaluated sample"
        )
    source = wer["per_sample"][0]
    translated_sample = translated["per_sample"][0]
    bleu = BLEU(effective_order=True)
    return {
        "asr_model": wer["config"]["asr_model"],
        "translation_model": translated["config"]["translation_model"],
        "arabic_wer": float(wer["summary"]["wer_corpus"]),
        "raw_arabic_bleu": bleu.sentence_score(
            source["whisper_text"], [source["target_text"]]
        ).score,
        "normalized_arabic_bleu": bleu.sentence_score(
            source["hyp_norm"], [source["ref_norm"]]
        ).score,
        "translated_english_wer": float(translated["summary"]["wer_corpus"]),
        "translated_english_bleu": bleu.sentence_score(
            translated_sample["translated_hyp"]["text"],
            [translated_sample["translated_ref"]["text"]],
        ).score,
        "reference_text": source["target_text"],
        "asr_text": source["whisper_text"],
        "translated_reference": translated_sample["translated_ref"]["text"],
        "translated_hypothesis": translated_sample["translated_hyp"]["text"],
    }


def main() -> None:
    args = _parse_args()
    summary = {
        "schema_version": 1,
        "sample_count": 1,
        "evaluated_wav_sha256": args.wav_sha256,
        "bleu_implementation": "sacrebleu BLEU effective_order=true",
        "pre_t1": _summarize(args.pre_t1_wer, args.pre_t1_translated_wer),
        "latest": _summarize(args.latest_wer, args.latest_translated_wer),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
