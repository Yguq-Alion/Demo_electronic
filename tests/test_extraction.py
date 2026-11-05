"""Толерантный разбор ответа модели: чистый JSON, заборы, преамбулы, мусор."""

from app.extraction import extract_json


def test_clean_json():
    assert extract_json('{"seo_title": "x"}') == {"seo_title": "x"}


def test_markdown_fence_json():  # I-13 модель ответила в ```json … ```
    assert extract_json('```json\n{"seo_title": "x"}\n```') == {"seo_title": "x"}


def test_bare_fence():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_preamble_before_json():  # преамбула «Вот карточка:»
    assert extract_json('Вот карточка:\n{"a": 1}') == {"a": 1}


def test_json_with_text_after():  # мусор после объекта
    assert extract_json('{"a": 1}\n\nГотово!') == {"a": 1}


def test_nested_braces():
    assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}


def test_garbage_returns_none():  # JSON отсутствует
    assert extract_json("Извините, не могу выполнить запрос") is None
    assert extract_json("") is None
    assert extract_json("```текст без объекта```") is None


def test_array_is_not_card():
    assert extract_json('[1, 2, 3]') is None
