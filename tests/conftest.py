"""Общие фикстуры: приложение с подменённой LLM + чистое окружение."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_llm


class FakeLlm:
    """Подменный LLM-клиент: возвращает заранее заданный сырой ответ."""

    def __init__(self, raw_response: str):
        self.raw_response = raw_response
        self.calls: list[tuple[str, str]] = []  # (system, user)

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.raw_response


@pytest.fixture
def client_factory():
    """TestClient с подменной LLM: client = client_factory(raw_llm_response);
    client_factory.last_fake — сам фейк (счётчик вызовов .calls)."""

    def make(raw_response: str) -> TestClient:
        fake = FakeLlm(raw_response)
        make.last_fake = fake
        app.dependency_overrides[get_llm] = lambda: fake
        return TestClient(app)

    make.last_fake = None
    yield make
    app.dependency_overrides.clear()


@pytest.fixture
def valid_product() -> dict:
    return {
        "name": "Манометр показывающий DM 02",
        "manufacturer": "Манотомь",
        "specs": {"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5"},
        "description": "Измерение избыточного давления.",
    }


VALID_CARD = {
    "seo_title": "Манометр показывающий DM 02, 0–1,6 МПа — Манотомь",
    "seo_description": "Манометр показывающий DM 02 от производителя Манотомь предназначен для измерения избыточного давления. Диапазон измерений 0–1,6 МПа, класс точности 1,5.",
    "specs": [
        {"name": "диапазон измерений", "value": "0–1,6 МПа"},
        {"name": "класс точности", "value": "1,5"},
        {"name": "питание", "value": "данных нет"},
    ],
    "search_queries": ["манометр Манотомь DM 02", "манометр показывающий 0–1,6 МПа", "манометр DM 02"],
    "commercial_block": "Прибор подойдёт службам, которым нужен контроль избыточного давления.",
    "missing_fields": [],
}


@pytest.fixture
def card_body() -> str:
    return json.dumps(VALID_CARD, ensure_ascii=False)
