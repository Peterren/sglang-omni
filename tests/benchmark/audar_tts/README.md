# Audar-TTS comparison runner

Run from the repository root on one CUDA GPU:

```bash
PYTHONPATH=. python tests/benchmark/audar_tts/run_pipeline_benchmark.py \
  --label <snapshot> \
  --output-dir /path/to/results \
  --repeats 7
```

The runner pins the Audar and NeuCodec revisions, reference sample, transcript,
target text, seed, and sampling settings. Exclude the first iteration when
comparing warm latency because it includes reference encoding and codec warmup.

The run also writes a SeedTTS-compatible `generated.json` under
`<output-dir>/seedtts/<snapshot>`. Score the saved audio with the repository's
ASR and WER pipeline:

```bash
OPENAI_API_KEY=<key> python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only \
  --model audarai/Audar-TTS-V1-Turbo \
  --output-dir /path/to/results/seedtts/<snapshot> \
  --lang ar \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --translated-wer \
  --translation-model gpt-5.6-luna \
  --translation-api-key-env OPENAI_API_KEY
```

`wer_results.json` is the primary Arabic ASR WER. The auxiliary
`translated_wer_results.json` translates the reference and ASR hypothesis in
separate API requests before reusing SeedTTS's English WER. Translations are
content-addressed in `translation_cache.json`, so reruns do not call the API.
