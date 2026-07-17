# Audar-TTS-V1 Turbo

The model weights use the
[AudarAI Community License](https://huggingface.co/audarai/Audar-TTS-V1-Turbo/blob/main/LICENSE).
Review its deployment terms before production use.

Install the optional GGUF and codec dependencies:

```bash
pip install -e '.[audar-tts]'
```

For a CUDA build of llama.cpp, install `llama-cpp-python` with the build flags
required by the target CUDA image before installing SGLang Omni.

Start the server with the explicit config because the Turbo Hugging Face repo
contains GGUF weights and no Transformers `config.json`:

```bash
sgl-omni serve --config examples/configs/audar_tts_turbo.yaml \
  --allowed-local-media-path /path/to/references
```

Send one 5-15 second reference clip and its transcript:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audarai/Audar-TTS-V1-Turbo",
    "input": "مرحبا، أهلا وسهلا بكم.",
    "language": "Arabic",
    "ref_audio": "file:///path/to/references/voice.wav",
    "ref_text": "النص المطابق للمقطع المرجعي.",
    "response_format": "wav"
  }' \
  --output audar.wav
```
