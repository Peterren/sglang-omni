# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, patch
from sglang_omni.client.audio import AudioStreamEncoder, encode_pcm


@pytest.mark.asyncio
async def test_audio_stream_encoder_wav_native_stream():
    encoder = AudioStreamEncoder(response_format="wav", sample_rate=24000)
    await encoder.start()

    # Generates a small float32 audio array (100 samples)
    audio = np.zeros(100, dtype=np.float32)

    # First chunk: Should contain WAV header + PCM data
    first_chunk = await encoder.encode_chunk(audio)
    assert len(first_chunk) > 200  # Header (44 bytes) + 16-bit PCM (200 bytes)
    assert first_chunk.startswith(b"RIFF")

    # Second chunk: Should contain ONLY raw PCM data (no header)
    second_chunk = await encoder.encode_chunk(audio)
    assert len(second_chunk) == 200
    assert second_chunk == encode_pcm(audio, sample_rate=24000)
    assert not second_chunk.startswith(b"RIFF")

    await encoder.finish()


@pytest.mark.asyncio
async def test_audio_stream_encoder_pcm_native_stream():
    encoder = AudioStreamEncoder(response_format="pcm", sample_rate=24000)
    await encoder.start()

    audio = np.zeros(100, dtype=np.float32)

    # Every chunk should be raw PCM data
    chunk = await encoder.encode_chunk(audio)
    assert len(chunk) == 200
    assert chunk == encode_pcm(audio, sample_rate=24000)

    await encoder.finish()


@pytest.mark.asyncio
async def test_audio_stream_encoder_ffmpeg_mp3_mocked():
    encoder = AudioStreamEncoder(response_format="mp3", sample_rate=24000)

    mock_process = AsyncMock()
    mock_process.stdin = AsyncMock()
    mock_process.stdout = AsyncMock()
    # Mock stdout buffer
    mock_process.stdout._buffer = bytearray(b"mocked-mp3-frame")
    mock_process.stdout.at_eof = lambda: False

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await encoder.start()
        assert mock_exec.called

        audio = np.zeros(100, dtype=np.float32)
        encoded = await encoder.encode_chunk(audio)

        # Should write to stdin and read from stdout
        assert mock_process.stdin.write.called
        assert mock_process.stdin.drain.called
        assert encoded == b"mocked-mp3-frame"

        # Finish should close stdin, wait, and read final bytes
        mock_process.stdout.read = AsyncMock(return_value=b"final-flushed-frame")
        final_encoded = await encoder.finish()
        assert mock_process.stdin.close.called
        assert mock_process.wait.called
        assert final_encoded == b"final-flushed-frame"
