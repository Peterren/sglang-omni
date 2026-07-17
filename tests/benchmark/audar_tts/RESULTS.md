# Audar-TTS pre/post T1 comparison

## Snapshots

- Pre-T1: `efad7215aaaf054d3597a4678e29e3370231b45a`, the unique parent of
  PR #807's squash commit.
- Current: `68187ff8bfd58d4442786817f79d90948324ff1b`, the `main` tip used for
  the final run.
- Pre-T1 benchmark implementation: `luojiaxuan/audar-tts-pre-t1` at
  `34d8ab7bba028bb79931042504405a1821549cd8`.
- Current benchmark implementation: `luojiaxuan/audar-tts-current` at
  `e223e68bafbf70de2045eff5870cc3711db07437`.
- Audar: `51f5635f32de3ab45ff28a4b958464532225b247`, Q4_K_M GGUF.
- NeuCodec: `30c1fdd19e68aee65d542cf043750d4c0165893e`.
- Hardware: one NVIDIA H100 80GB, CUDA 13.0, PyTorch 2.11.0, and
  llama-cpp-python 0.3.34 with CUDA enabled and NCCL disabled.

The summarized metrics in this document and `comparison.json` are the Git
source of truth. Raw run outputs remain local-only at
`/data/jaxan/audar-results` on H100 and have not been uploaded.

## Correctness

Both snapshots produced the same 285 acoustic codes and the same float32
waveform for all 14 measured requests per snapshot:

- Acoustic-code SHA-256: `f9f63e0ca99e82bf7bbcab20a3b6d6fc7e36bc73a320f2f84a7942edde6bc98e`
- Waveform SHA-256: `44ec732c1243bf5d6115783b318de2cd7691f5ba0ab6a75c1e01ea85eae73492`
- HTTP WAV SHA-256: `66128571a5a473ababa0bb2c6bc7a5ed1ee625cf580979ff6c906b1f2aa64b4c`
- Output: 24 kHz mono, 5.7 seconds.

Whisper large-v3-turbo at revision
`41f01f3fe87f28c78e2fbf8b568835947dd65ed9` transcribed the output as the
exact target after Arabic normalization: CER 0.0 and WER 0.0.

The repository's SeedTTS pipeline was also run with Qwen3-ASR-1.7B at revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`. It produced the same normalized
Arabic reference and hypothesis for both snapshots, with corpus WER 0.0. The
measured ASR request latency was 95.8 ms for current and 94.8 ms for pre-T1;
these one-sample timings validate the evaluator and are not model-performance
comparisons.

GPT-5.6 Luna translated the reference and ASR hypothesis independently to the
same English sentence: "We are pleased to test today the system for converting
Arabic text into clear and natural speech." Translated English WER was 0.0 for
both snapshots. The current run made two API calls using 277 tokens (217 input,
60 output); pre-T1 reused both content-addressed cache entries and made no API
calls. This one-sample translated score is auxiliary because translation can
hide Arabic ASR differences; Arabic WER remains the primary correctness metric.

The benchmark initially found that a persistent llama.cpp instance changed its
first result despite a fixed seed. Explicitly resetting llama.cpp before each
request fixed request isolation; the unit test now enforces that reset.

## Performance

Two seven-request runs were executed in each order on an otherwise empty GPU.
The table averages each snapshot's warm median from both orders; iteration zero
is excluded.

| Snapshot | Total latency | RTF | Engine latency | Engine codes/s |
| --- | ---: | ---: | ---: | ---: |
| Pre-T1 | 1.0773 s | 0.1890 | 1.0688 s | 294.23 |
| Current | 1.1021 s | 0.1934 | 1.0924 s | 288.23 |

Current measured 2.3% slower end to end. The generation implementation is the
same in both branches and run-to-run variation was larger than this delta, so
this is not evidence of an architecture-level performance regression or gain.
The serial GGUF decode also stayed below 90% average GPU utilization; one GPU
is already the minimum allocation.

## Integration size

Git added-line counts use each snapshot as its own baseline and exclude all
`tests/**` files.

| Scope | Pre-T1 | Current | Delta |
| --- | ---: | ---: | ---: |
| Non-test, including model README | 673 | 731 | +58 (+8.6%) |
| Code/config, excluding README | 635 | 693 | +58 (+9.1%) |
| Model-package Python only | 625 | 683 | +58 (+9.3%) |

The current landed architecture does not reduce integration code for this
model. `PipelineStateBase` and `BatchVocoderBase` remove local mechanics, but
the `ReferenceEncodeHook` adapter and `ModelCapabilities` declaration cost more
lines than they save. The shared service does add single-flight deduplication,
path revalidation, failure fan-out, and cache statistics that the pre-T1
implementation lacks.

PR #1050 is not in the measured current snapshot. Declarative state fields and
a lower-boilerplate functional reference-encoder adapter are the two concrete
changes most likely to turn the integration-size result negative.
