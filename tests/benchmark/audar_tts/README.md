# Audar-TTS comparison runner

Run the paired comparison from the latest checkout on one CUDA GPU. The driver
uses the fixed order latest, pre-T1, pre-T1, latest and writes the aggregate
summary automatically:

```bash
python tests/benchmark/audar_tts/compare_pipeline_benchmarks.py \
  --pre-t1-checkout /path/to/pre-t1-checkout \
  --latest-checkout /path/to/latest-checkout \
  --output-dir /path/to/results \
  --repeats 7
```

To run one snapshot directly from its repository root:

```bash
PYTHONPATH=. python tests/benchmark/audar_tts/run_pipeline_benchmark.py \
  --label <snapshot> \
  --output-dir /path/to/results \
  --repeats 7
```

The runner pins the Audar and NeuCodec revisions, reference sample, transcript,
target text, seed, and sampling settings. `warm_summary` excludes iteration zero,
which includes reference encoding and codec warmup.

For Arabic intelligibility, score the saved WAV with the repository SeedTTS ASR
pipeline. Translate the reference and ASR hypothesis independently before
computing English metrics; do not translate only one side.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only \
  --model audarai/Audar-TTS-V1-Turbo \
  --output-dir /path/to/seedtts/results \
  --lang ar \
  --asr-model-path Qwen/Qwen3-ASR-1.7B
```

Arabic WER is the primary quality signal. Translated English WER/BLEU is an
auxiliary diagnostic because translation can hide Arabic recognition errors.
See [RESULTS.md](./RESULTS.md) for the locked comparison.

Regenerate the committed text-metric summary from the raw ASR artifacts:

```bash
python tests/benchmark/audar_tts/summarize_quality.py \
  --pre-t1-wer tests/benchmark/audar_tts/artifacts/quality/pre-t1-wer-results.json \
  --pre-t1-translated-wer tests/benchmark/audar_tts/artifacts/quality/pre-t1-translated-wer-results.json \
  --latest-wer tests/benchmark/audar_tts/artifacts/quality/latest-wer-results.json \
  --latest-translated-wer tests/benchmark/audar_tts/artifacts/quality/latest-translated-wer-results.json \
  --wav-sha256 e7f0b6bb3ea5d950ff2fc1329a63954c2579cf97201dafb0a61514eb9b5ca04b \
  --output tests/benchmark/audar_tts/artifacts/quality/quality_summary.json
```
