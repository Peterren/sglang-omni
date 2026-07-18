# Audar-TTS pre/post refactor comparison

## Locked snapshots

- Pre-T1 baseline: `efad7215aaaf054d3597a4678e29e3370231b45a`, the parent
  of PR #807.
- Latest baseline: `98b634332517ad2c9a88ff7f96880aae251a375c`, the current
  `main` tip on 2026-07-18 and merge commit for PR #1070.
- Pre-T1 production integration:
  `ec25335e9815ef53daa9c60239a1859804282270`.
- Latest production integration:
  `49cde765101bf3c4501dc1838f47ec23dbbf225b`.
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
| Total latency | 1.0779 s | 1.0764 s | -0.13% |
| RTF | 0.1891 | 0.1888 | -0.13% |
| Engine wall latency | 1.0696 s | 1.0685 s | -0.11% |
| Engine codes/s | 293.95 | 294.43 | +0.16% |
| Cached reference stage | 0.368 ms | 0.369 ms | +0.001 ms |
| Vocoder | 7.903 ms | 7.463 ms | -5.56% |

The latest integration is 0.13% faster in this sample. The two pre-T1 run
medians were 1.0749 and 1.0808 seconds, wider than the pre/post difference, so
the result is performance parity rather than evidence of a speedup or
regression. The paired driver fixes the order to latest, pre-T1, pre-T1,
latest. The AR implementation is the same serial llama.cpp decode in both
branches.

Steady-state H100 SM utilization was 61-64%. One GPU is already the minimum
allocation; this single-request latency workload is limited by serial GGUF
decoding and Python token handling rather than available GPU count.

## Integration size

Tests and Markdown/RST files are excluded from this LOC table.

| Capability tier | Pre-T1 | Latest | Reduction |
| --- | ---: | ---: | ---: |
| Minimal integration | 575 | 543 | 32 (5.6%) |
| Production-enhanced integration | 797 | 633 | 164 (20.6%) |
| Production capability premium | 222 | 90 | 132 (59.5%) |

The production tier includes bounded reference caching, composite keys,
same-reference single-flight, distinct-reference request isolation, serialized
NeuCodec forwards, failure fan-out, path revalidation, and cache statistics.
NeuCodec does not expose a tensor-level batch decode API here, so neither side
claims vocoder batching or adds a batch wait.

## Source of truth

- Lightweight summary: this file and `comparison.json` in Git.
- Raw run outputs: `/data/jaxan/audar-results-production-equal-final` on H100.
- Prior ASR/translation outputs:
  `/data/jaxan/audar-results/seedtts` on H100.
- Raw artifacts are local-only. Intended Hugging Face dataset destination is
  pending a credential for the correct owner; the shared H100 credential belongs
  to another user and was not used for upload.
