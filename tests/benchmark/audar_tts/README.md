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
