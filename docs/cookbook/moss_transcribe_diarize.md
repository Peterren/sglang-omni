# MOSS-Transcribe-Diarize

[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) is a multi-speaker ASR and diarization model from the OpenMOSS team. It transcribes dialog audio with speaker labels and timestamps, and is served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint.

【TODO：这个应该重点强调 long sequence + multi-speaker 的性质吧 @gaoyang】

| Component | Spec |
|---|---|
| Architecture | `MossTranscribeDiarizeForConditionalGeneration` |
| Audio encoder | Whisper encoder (24 L, d_model=1024) |
| Text decoder | Qwen3 (28 L, hidden=1024, GQA 16/8) |
| Output | Speaker-labelled transcript with start/end timestamps |
| Endpoint | `/v1/audio/transcriptions` |

## Model Usage

### Launching Commands

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize
```

Serve the model:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --port 8000 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80
```

### Sending Requests

Use `response_format=verbose_json` when you need parsed speaker segments. `json` returns the raw transcript text only.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=verbose_json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "response_format": "verbose_json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
payload = resp.json()
print(payload["text"])
for segment in payload.get("segments", []):
    print(
        f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}"
    )
```

For longer multi-speaker audio, raise `max_new_tokens` so the decoder can finish the full diarized transcript. The example below uses a repo-local clip with two speakers:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@docs/_static/audio/gaokao-listening.wav \
  -F response_format=verbose_json \
  -F max_new_tokens=65536
```

### Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional language hint |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | model default (`0.0`) | Sampling temperature |
| `max_new_tokens` | int | `5120` | Max generated tokens; raise for long audio (e.g. `65536`) |
| `prompt` | string | unset | Optional instruction override; omit to use the built-in transcribe+diarize prompt |

`verbose_json` parses the model markup into OpenAI-style `segments` with
`start`, `end`, and speaker-prefixed `text` (for example `[S01]...`).
`json` / `text` return the full transcript string without segment parsing.


## Performance Optimization

【TODO：@yichi】

## Benchmarking

Thanks to the Moss team for providing the benchmark datasets, we prepare movies800times and aishell4_long as benchmark datasets for multi-speaker ASR. movies800times is a short-sequence dataset with 800 dialog clips, and aishell4_long is a long-sequence dataset with 20 long-form meeting audio. These two datasets are right now under private license, and you can contact the Moss team for access.


```bash
# Short-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times \
  --concurrency 16 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80 \
  --output-dir results/moss_transcribe_diarize_movies800times

# Long-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset aishell4_long \
  --concurrency 16 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80 \
  --max-new-tokens 65536 \
  --request-timeout-s 1800 \
  --output-dir results/moss_transcribe_diarize_aishell4_long
```

## Benchmark Results

Here we provide the benchmark results of movies800times and aishell4_long on a single H100 80GB GPU.

### movies800times

| Concurrency | Throughput (req/s) | Mean latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 2.43 | 0.411 | 0.0663 | 28.12 |
| 2 | 4.53 | 0.441 | 0.0729 | 52.43 |
| 4 | 6.88 | 0.507 | 0.0792 | 79.66 |
| 8 | 7.65 | 0.504 | 0.0747 | 88.53 |
| 16 | 6.58 | 0.660 | 0.0933 | 76.18 |

### aishell4_long

| Concurrency | Throughput (req/s) | Mean latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 0.022 | 45.4 | 0.0198 | 50.49 |
| 2 | 0.032 | 62.1 | 0.0271 | 72.56 |
| 4 | 0.034 | 112.2 | 0.0490 | 77.01 |
| 8 | 0.039 | 175.0 | 0.0765 | 89.40 |
| 16 | 0.040 | 308.6 | 0.1348 | 91.63 |


- **Concurrency** — Maximum number of in-flight client requests (`--concurrency`).
- **Throughput (req/s)** — Completed requests divided by total benchmark wall-clock time.
- **Mean latency** — Average end-to-end time per request (send to full response received).
- **RTF mean** — Average ratio of processing time to input audio duration per request. `<1` is faster than real time.
- **audio_s/s** — Total seconds of input audio processed divided by total benchmark wall-clock time.

To reproduce the results, follow the commands above or the entry point in [`benchmark_asr_transcribe_diarize.py`](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_asr_transcribe_diarize.py).
