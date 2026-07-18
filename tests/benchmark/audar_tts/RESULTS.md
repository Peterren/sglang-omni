# Audar-TTS pre/post refactor comparison

## Locked snapshots

- Pre-T1 baseline: `efad7215aaaf054d3597a4678e29e3370231b45a`, the parent
  of PR #807.
- Latest baseline: `98b634332517ad2c9a88ff7f96880aae251a375c`, the current
  `main` tip on 2026-07-18 and merge commit for PR #1070.
- Pre-T1 production integration:
  `56714392f7d982ce0ce294ef39a547362b594132`.
- Latest production integration:
  `b22341e0d8dd7a3b90fc169713c92de3665e0079`.
- Audar revision: `51f5635f32de3ab45ff28a4b958464532225b247`.
- NeuCodec revision: `30c1fdd19e68aee65d542cf043750d4c0165893e`.
- Runtime: one NVIDIA H100 80GB, driver 580.126.20, CUDA 13.0,
  PyTorch 2.11.0, and CUDA-enabled llama-cpp-python 0.3.34.

## Correctness and quality

Both integrations generated the same 285 acoustic codes and the same 136,800
float32 waveform samples on all 28 requests. The output is 5.7 seconds of
24 kHz mono audio.

- Acoustic-code SHA-256:
  `f9f63e0ca99e82bf7bbcab20a3b6d6fc7e36bc73a320f2f84a7942edde6bc98e`
- Float waveform SHA-256:
  `44ec732c1243bf5d6115783b318de2cd7691f5ba0ab6a75c1e01ea85eae73492`
- PCM WAV SHA-256:
  `e7f0b6bb3ea5d950ff2fc1329a63954c2579cf97201dafb0a61514eb9b5ca04b`

The WAV hash is also identical to the earlier file evaluated by Qwen3-ASR-1.7B
and GPT-5.6 Luna. Reusing that content-addressed evaluation gives:

| Metric | Pre-T1 | Latest | Notes |
| --- | ---: | ---: | --- |
| Arabic WER | 0.0 | 0.0 | Qwen3-ASR-1.7B, normalized Arabic |
| Arabic CER | 0.0 | 0.0 | Whisper large-v3-turbo, normalized Arabic |
| Translated English WER | 0.0 | 0.0 | GPT-5.6 Luna translations |
| Translated English BLEU | 100.0 | 100.0 | SacreBLEU 2.6, effective order |
| Raw Arabic BLEU | 91.22 | 91.22 | Only diacritic difference in ASR text |
| Normalized Arabic BLEU | 100.0 | 100.0 | Same normalization as WER |

This is a one-sentence regression check, not a statistically meaningful model
quality benchmark. BLEU is secondary for TTS; Arabic WER/CER and exact waveform
equality are the stronger correctness signals here. Speaker similarity, MOS,
and UTMOS were not measured in this run.

## Performance

Each snapshot ran twice in alternating order with seven requests per run.
Iteration zero was excluded. The table averages the warm median of each run,
so each snapshot contributes 12 measured warm requests.

| Metric | Pre-T1 | Latest | Delta |
| --- | ---: | ---: | ---: |
| Total latency | 1.0744 s | 1.0807 s | +0.59% |
| RTF | 0.1885 | 0.1896 | +0.59% |
| Engine wall latency | 1.0663 s | 1.0728 s | +0.60% |
| Engine codes/s | 295.16 | 293.34 | -0.62% |
| Cached reference stage | 0.349 ms | 0.383 ms | +0.034 ms |
| Vocoder | 7.580 ms | 7.466 ms | -1.50% |

The latest integration is 0.59% slower in this sample. The two latest run
medians were 1.0778 and 1.0835 seconds, wider than the pre/post difference, so
this is not evidence of a performance regression. The AR implementation is the
same serial llama.cpp decode in both branches.

Steady-state H100 SM utilization was 61-63%. One GPU is already the minimum
allocation; this workload is limited by serial GGUF decoding and Python token
handling rather than available GPU count.

## Integration size

Tests and Markdown/RST files are excluded from this LOC table.

| Capability tier | Pre-T1 | Latest | Reduction |
| --- | ---: | ---: | ---: |
| Minimal integration | 575 | 543 | 32 (5.6%) |
| Production-enhanced integration | 820 | 656 | 164 (20.0%) |
| Production capability premium | 245 | 113 | 132 (53.9%) |

The production tier includes bounded reference caching, composite keys,
same-reference single-flight, different-reference concurrency, failure fan-out,
path revalidation, cache statistics, and vocoder batching.

## Source of truth

- Lightweight summary: this file and `comparison.json` in Git.
- Raw run outputs: `/data/jaxan/audar-results-latest-prod-comparison` on H100.
- Prior ASR/translation outputs:
  `/data/jaxan/audar-results/seedtts` on H100.
- Raw artifacts are local-only. Intended Hugging Face dataset destination is
  pending a credential for the correct owner; the shared H100 credential belongs
  to another user and was not used for upload.
