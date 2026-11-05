"""Кейсы X-категории (невалидный вход) + граничные E: код отвечает 400
до вызова LLM, тексты ошибок совпадают с промптом."""

import json

import pytest

from app.validation import parse_and_validate

PRODUCT = {"name": "Манометр DM 02", "manufacturer": "Манотомь", "specs": {"класс точности": "1,5"}, "description": "Измерение давления."}


@pytest.mark.parametrize("raw, expected_error", [
    ('{"name": "Манометр", "specs": {', "Невалидный вход: ожидался JSON-объект товара"),  # X-01 обрезанный JSON
    ("привет", "Невалидный вход: ожидался JSON-объект товара"),  # X-10 не JSON вообще
    ("", "Невалидный вход: ожидался JSON-объект товара"),
    ("42", "Невалидный вход: ожидается объект товара, а не массив или число"),  # X-10 число
    ('[{"name": "Манометр"}]', "Невалидный вход: ожидается объект товара, а не массив или число"),  # X-02 массив
    ('{}', "Невалидный вход: не хватает обязательного поля name"),  # X-05 пустой объект
    ('{"name": "", "specs": {}}', "Невалидный вход: не хватает обязательного поля name"),  # E-05 пустой name
    ('{"name": "   ", "specs": {}}', "Невалидный вход: не хватает обязательного поля name"),  # E-03 пробелы
    ('{"manufacturer": "Манотомь"}', "Невалидный вход: не хватает обязательного поля name"),  # name нет
    ('{"name": "Манометр", "specs": "0-1,6 МПа"}', "Невалидный вход: поле specs должно быть объектом «характеристика: значение»"),  # X-03
    ('{"name": "Манометр", "specs": ["a"]}', "Невалидный вход: поле specs должно быть объектом «характеристика: значение»"),
    ('{"name": "Манометр", "specs": {"диапазон": {"from": 0}}}', "Невалидный вход: значения характеристик должны быть строкой или числом"),  # X-07 вложенный объект
    ('{"name": "Манометр", "specs": {"диапазон": [0, 1.6]}}', "Невалидный вход: значения характеристик должны быть строкой или числом"),
    ('{"name": "Манометр", "specs": {"флаг": true}}', "Невалидный вход: значения характеристик должны быть строкой или числом"),  # bool не число
    ('{"name": "Манометр", "description": 12345}', "Невалидный вход: поле description должно быть строкой"),  # X-04
    ('{"name": "Манометр", "description": ["текст"]}', "Невалидный вход: поле description должно быть строкой"),
])
def test_invalid_input_rejected_before_llm(raw, expected_error):
    product, error = parse_and_validate(raw)
    assert product is None
    assert error == expected_error


def test_valid_minimal_product_passes():
    product, error = parse_and_validate('{"name": "Манометр DM 02"}')
    assert error is None
    assert product["name"] == "Манометр DM 02"


def test_number_spec_value_allowed():  # E-12 число не строкой — валидно
    product, error = parse_and_validate('{"name": "М", "specs": {"класс точности": 1.5}}')
    assert error is None
    assert product["specs"]["класс точности"] == 1.5


def test_unknown_top_level_fields_kept():  # X-06 price/sku не ломают вход
    payload = dict(PRODUCT, price=1500, sku="X-1")
    product, error = parse_and_validate(json.dumps(payload))
    assert error is None
    assert product["price"] == 1500


def test_long_description_cut_to_40000():  # E-07 экономия токенов
    payload = dict(PRODUCT, description="а" * 55_000)
    product, error = parse_and_validate(json.dumps(payload))
    assert error is None
    assert len(product["description"]) == 40_000
