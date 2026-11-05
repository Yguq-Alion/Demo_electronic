"""Оркестрация генерации карточки: промпт → LLM → разбор ответа.

Семантическая ошибка (модель вернула {"error": ...}) — не сбой: это валидный
вердикт по входу (например, «товар не относится к КИП»), отдаётся вызывающему
отдельным исключением с кодом 422.
"""

from .extraction import extract_json
from .llm import LlmClient, LlmError
from .prompts import SYSTEM_PROMPT, build_user_message
from .schemas import CARD_KEYS, ProductCard


class SemanticError(Exception):
    """Модель отклонила вход по смыслу (не-КИП и т.п.). Текст — для ответа 422."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def generate_card(product: dict, llm: LlmClient) -> ProductCard:
    user_message = build_user_message(product)
    raw = llm.chat(SYSTEM_PROMPT, user_message)

    data = extract_json(raw)
    if data is None:
        raise LlmError("Модель вернула ответ без JSON — повторите запрос или смените LLM_MODEL", 502)

    if "error" in data:
        raise SemanticError(str(data["error"]) or "Модель отклонила входные данные")

    missing = CARD_KEYS - data.keys()
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise LlmError(f"Модель вернула карточку без секций: {missing_list}", 502)

    try:
        return ProductCard.model_validate(data)
    except Exception:
        raise LlmError("Модель вернула карточку с неверными типами полей", 502)
