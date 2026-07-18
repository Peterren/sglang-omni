# Audar-TTS comparison runner

Run from the repository root on one CUDA GPU:

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

Arabic WER/CER is the primary quality signal. Translated English WER/BLEU is an
auxiliary diagnostic because translation can hide Arabic recognition errors.
See [RESULTS.md](./RESULTS.md) for the locked comparison.
