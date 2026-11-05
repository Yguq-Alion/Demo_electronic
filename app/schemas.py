"""Pydantic-модели ответа. Поля не строже контракта промпта: лимиты длин
(70 / 800–1200 / 3–5) enforcing'ся промптом и проверяются прогоном оценок,
жёсткая схема здесь роняла бы валидные по сути ответы из-за мелких отступлений.
"""

from pydantic import BaseModel


class SpecItem(BaseModel):
    name: str
    value: str


class ProductCard(BaseModel):
    seo_title: str
    seo_description: str
    specs: list[SpecItem]
    search_queries: list[str]
    commercial_block: str
    missing_fields: list[str]


CARD_KEYS = frozenset(ProductCard.model_fields)
