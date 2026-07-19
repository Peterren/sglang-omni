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
- Quality artifact heads: pre-T1 `44eaf8819f34723d4ad132045614685a389d5f15`
  and latest `80d9b2a4062247406cd56ff43aa9f3e1e4b4cf3a`. Changes after the
  production integration commits only add benchmark evidence, tests, docs, and
  a latest-side import-order cleanup; model behavior is unchanged.
- Audar revision: `51f5635f32de3ab45ff28a4b958464532225b247`.
- NeuCodec revision: `30c1fdd19e68aee65d542cf043750d4c0165893e`.
- Runtime: one NVIDIA H100 80GB, driver 580.126.20, CUDA 13.0,
  PyTorch 2.11.0, and CUDA-enabled llama-cpp-python 0.3.34.

## Correctness

Both integrations generated the same 285 acoustic codes and the same 136,800
float32 waveform samples on all 28 requests. The output is 5.7 seconds of
24 kHz mono audio.

- Acoustic-code SHA-256:
  `f9f63e0ca99e82bf7bbcab20a3b6d6fc7e36bc73a320f2f84a7942edde6bc98e`
- Float waveform SHA-256:
  `44ec732c1243bf5d6115783b318de2cd7691f5ba0ab6a75c1e01ea85eae73492`
- PCM WAV SHA-256:
  `e7f0b6bb3ea5d950ff2fc1329a63954c2579cf97201dafb0a61514eb9b5ca04b`

## Quality

The quality run uses 50 texts from the pinned FLEURS `ar_eg` test Parquet. It
keeps rows from the explicit `transcription` column with 6-20 words, at least
90% Arabic letters, and no digits. Digit rows are excluded because ASR
verbalizes numbers while references retain digits. The target text is the
reference; Qwen3-ASR-1.7B transcribes the speech. Metrics are computed directly
between normalized Arabic references and Arabic ASR hypotheses. No translation
is used.

All 50 pre-T1 and latest acoustic-code, float-waveform, and PCM-WAV hashes match
pairwise. The ASR artifact also records and verifies the SHA-256 of every WAV it
actually transcribed. Latest was transcribed once; the pre-T1 quality result is
inherited only from exact WAV identity, not measured in a second ASR run.

| Audar TTS metric | Value |
| --- | ---: |
| Arabic corpus WER | 5.43% |
| Arabic corpus CER | 1.46% |
| Arabic corpus BLEU | 88.75 |
| Arabic chrF++ | 95.57 |

The same ASR and normalization give 8.91% WER on the original human FLEURS
audio for these rows. This is context for the recognizer, not a directly
comparable floor or quality delta: the source audio can diverge from the written
transcription, including spoken-number and grammatical-form differences, and it
is 16 kHz rather than the TTS output's 24 kHz. No sample has WER above 50%;
maximum per-sample WER is 35.29% for TTS and 37.50% for source audio.

BLEU, CER, and chrF++ are alternate views of the same ASR transcripts, not
independent quality measurements. Hash identity alone establishes pre/post
refactor equivalence; these ASR metrics are a separate absolute-intelligibility
check for Audar.
These values come from one Qwen3-ASR pass, so no ASR run-to-run interval is
claimed. Metrics use folded Arabic orthography: Alef variants and alif maqsura
are normalized, and diacritics, tatweel, and punctuation are removed. They
measure intelligibility, not naturalness or speaker similarity. MOS, UTMOS, and
speaker similarity were not measured.

## Performance

Each snapshot ran twice in alternating order with seven requests per run.
Iteration zero was excluded. The table averages the warm median of each run,
so each snapshot contributes 12 measured warm requests.

| Metric | Pre-T1 | Latest | Delta |
| --- | ---: | ---: | ---: |
| Stage-sum latency | 1.0779 s | 1.0764 s | -0.13% |
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

The quality generator is not a performance benchmark: it runs pre-T1 first and
has no warmup exclusion. Its first two pre-T1 blocks ramp from 157.57 to 193.80
codes/s. Over samples 20-49, after that startup effect, pre-T1 and latest are
274.78 and 274.02 codes/s. The balanced performance driver above excludes
warmup and is the source of the parity conclusion.

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
- Machine-generated evidence: `artifacts/performance/performance_summary.json`,
  `artifacts/performance/wav_sha256s.txt`, and the generation, ASR, and summary
  JSON files under `artifacts/quality/`.
- Raw run outputs: `/data/jaxan/audar-results-production-equal-final` on H100.
- Raw 50-sentence quality outputs:
  `/data/jaxan/audar-quality-results-fleurs50-nodigits-v2` on H100.
- Raw artifacts are local-only. Intended Hugging Face dataset destination is
  pending a credential for the correct owner; the shared H100 credential belongs
  to another user and was not used for upload.
