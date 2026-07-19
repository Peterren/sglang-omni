# Audar-TTS comparison runner

Run the paired comparison from the latest checkout on one CUDA GPU. The driver
uses the fixed order latest, pre-T1, pre-T1, latest and writes the aggregate
summary automatically:

```bash
python benchmarks/audar_tts/compare_pipeline_benchmarks.py \
  --pre-t1-checkout /path/to/pre-t1-checkout \
  --latest-checkout /path/to/latest-checkout \
  --output-dir /path/to/results \
  --repeats 7
```

To run one snapshot directly from its repository root:

```bash
PYTHONPATH=. python benchmarks/audar_tts/run_pipeline_benchmark.py \
  --label <snapshot> \
  --output-dir /path/to/results \
  --repeats 7
```

The runner pins the Audar and NeuCodec revisions, reference sample, transcript,
target text, seed, and sampling settings. `warm_summary` excludes iteration zero,
which includes reference encoding and codec warmup.

For Arabic intelligibility, generate the pinned digit-free 50-sentence FLEURS
`ar_eg` set from both checkouts. The same script runs against each checkout through
`PYTHONPATH`, so text selection and generation settings are identical:

```bash
python benchmarks/audar_tts/compare_quality_benchmarks.py \
  --pre-t1-checkout /path/to/pre-t1-checkout \
  --latest-checkout /path/to/latest-checkout \
  --output-dir /path/to/quality-results \
  --samples 50
```

Materialize the original FLEURS audio for the same selected rows:

```bash
python benchmarks/audar_tts/prepare_fleurs_reference_audio.py \
  --output-dir /path/to/quality-results/reference \
  --samples 50
```

Transcribe the latest WAVs directly to Arabic with the repository SeedTTS ASR
pipeline. When all paired waveform hashes match, this one ASR run is also the
pre-T1 evidence. In `--transcribe-only` mode, the benchmark reads the existing
`generated.json`; `--meta` is recorded as provenance and does not reselect rows:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only \
  --model audarai/Audar-TTS-V1-Turbo \
  --output-dir /path/to/quality-results/latest \
  --meta google/fleurs \
  --lang ar \
  --max-new-tokens 1024 \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --asr-concurrency 16 \
  --skip-gpu-cleanup
```

Run the same command for `/path/to/quality-results/reference`, changing
`--model` to `google/fleurs-ar-eg-reference-audio`. This provides the ASR
baseline on the original human recordings.

The TTS target text is the Arabic reference. Do not translate either side.
`summarize_quality.py` computes normalized Arabic corpus BLEU, WER, CER, and
chrF++ from the target and Arabic ASR hypothesis. See [RESULTS.md](./RESULTS.md)
for the locked comparison.

Regenerate the committed text-metric summary from the raw ASR artifacts:

```bash
python benchmarks/audar_tts/summarize_quality.py \
  --pre-t1-generation benchmarks/audar_tts/artifacts/quality/pre-t1-generation.json \
  --latest-generation benchmarks/audar_tts/artifacts/quality/latest-generation.json \
  --latest-wer benchmarks/audar_tts/artifacts/quality/latest-wer-results.json \
  --reference-audio-generation benchmarks/audar_tts/artifacts/quality/reference-audio-generation.json \
  --reference-wer benchmarks/audar_tts/artifacts/quality/reference-wer-results.json \
  --output benchmarks/audar_tts/artifacts/quality/quality_summary.json
```
