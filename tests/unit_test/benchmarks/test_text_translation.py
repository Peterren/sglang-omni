from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.tasks import tts
from benchmarks.tasks.text_translation import (
    TranslationBatch,
    TranslationRecord,
    translate_arabic_texts_to_english,
)


class _FakeResponses:
    def __init__(self, translations: list[str]) -> None:
        self.translations = iter(translations)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        translation = next(self.translations)
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            model="gpt-5.6-luna-resolved",
            output_text=json.dumps({"translation": translation}),
            usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )


class _FakeClient:
    def __init__(self, translations: list[str]) -> None:
        self.responses = _FakeResponses(translations)


def test_arabic_normalization_for_seedtts_wer() -> None:
    assert tts.normalize_text("إِنَّ الـنَّصَّ، رقم ١٢!", "ar") == "ان النص رقم 12"


def test_translation_requests_are_isolated_and_cached(tmp_path) -> None:
    cache_path = tmp_path / "translations.json"
    client = _FakeClient(["We are pleased.", "We are happy."])
    source = ["يسرنا.", "نحن سعداء."]

    first = translate_arabic_texts_to_english(
        source,
        cache_path=cache_path,
        client=client,
    )

    assert [record.translated_text for record in first.records] == [
        "We are pleased.",
        "We are happy.",
    ]
    assert first.api_calls == 2
    assert first.cache_hits == 0
    assert len(client.responses.calls) == 2
    assert source[0] in client.responses.calls[0]["input"]
    assert source[1] not in client.responses.calls[0]["input"]
    assert source[1] in client.responses.calls[1]["input"]
    assert source[0] not in client.responses.calls[1]["input"]
    assert client.responses.calls[0]["store"] is False
    assert client.responses.calls[0]["temperature"] == 0
    assert client.responses.calls[0]["text"]["format"]["strict"] is True

    second = translate_arabic_texts_to_english(source, cache_path=cache_path)

    assert second.api_calls == 0
    assert second.cache_hits == 2
    assert [record.translated_text for record in second.records] == [
        "We are pleased.",
        "We are happy.",
    ]


def test_translation_requires_key_for_cache_miss(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TEST_OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TEST_OPENAI_API_KEY is not set"):
        translate_arabic_texts_to_english(
            ["مرحبا"],
            api_key_env="TEST_OPENAI_API_KEY",
            cache_path=tmp_path / "translations.json",
        )


def test_translated_wer_reuses_seedtts_english_metric(tmp_path, monkeypatch) -> None:
    source = tts.SampleOutput(
        sample_id="sample-1",
        target_text="مرحبا بالعالم",
        whisper_text="مرحبا هناك",
        wer=0.5,
        is_success=True,
    )
    translated_records = [
        TranslationRecord(
            source_text=source.target_text,
            translated_text="hello world",
            cache_key="ref",
            cache_hit=False,
            requested_model="gpt-test",
            response_model="gpt-test-resolved",
            response_id="response-ref",
            usage={"total_tokens": 10},
        ),
        TranslationRecord(
            source_text=source.whisper_text,
            translated_text="hello there",
            cache_key="hyp",
            cache_hit=False,
            requested_model="gpt-test",
            response_model="gpt-test-resolved",
            response_id="response-hyp",
            usage={"total_tokens": 11},
        ),
    ]
    monkeypatch.setattr(
        tts,
        "translate_arabic_texts_to_english",
        lambda *args, **kwargs: TranslationBatch(
            records=translated_records,
            cache_path=str(tmp_path / "translations.json"),
            cache_hits=0,
            api_calls=2,
        ),
    )
    config = SimpleNamespace(
        output_dir=str(tmp_path),
        translation_model="gpt-test",
        translation_api_key_env="TEST_OPENAI_API_KEY",
        translation_cache=None,
    )

    result = tts._translated_wer_results([source], config, {})

    assert result["summary"]["wer_corpus"] == 0.5
    assert result["summary"]["source_lang"] == "ar"
    assert result["translation"]["api_calls"] == 2
    assert result["translation"]["usage"]["total_tokens"] == 21
    assert result["per_sample"][0]["translated_ref"]["text"] == "hello world"
