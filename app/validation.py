"""Сервисная валидация входа кодом — до вызова LLM.

Правила 1–6 дословно повторяют раздел «Валидация входа» системного промпта
(двойная защита: код отвечает мгновенно и без расхода токенов, модель
дублирует те же проверки как последнюю линию и добавляет семантическую
проверку «товар относится к КИП», которую кодом не сделать).
Тексты ошибок совпадают с промптом, чтобы поведение было единым.

None = вход структурно корректен; строка = текст ошибки для ответа 400.
"""

import json

MAX_DESCRIPTION_CHARS = 40_000  # промпт читает первые 40000 символов — режем заранее, экономия токенов


def parse_and_validate(raw: str) -> tuple[dict | None, str | None]:
    """Разбор тела запроса и структурная валидация. Возвращает (товар, ошибка)."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, "Невалидный вход: ожидался JSON-объект товара"

    if not isinstance(payload, dict):
        return None, "Невалидный вход: ожидается объект товара, а не массив или число"

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, "Невалидный вход: не хватает обязательного поля name"

    specs = payload.get("specs")
    if specs is not None and not isinstance(specs, dict):
        return None, "Невалидный вход: поле specs должно быть объектом «характеристика: значение»"

    if isinstance(specs, dict):
        for value in specs.values():
            # bool в Python — подкласс int; булево значение не считается числом
            if isinstance(value, (dict, list)) or isinstance(value, bool):
                return None, "Невалидный вход: значения характеристик должны быть строкой или числом"

    description = payload.get("description")
    if description is not None and not isinstance(description, str):
        return None, "Невалидный вход: поле description должно быть строкой"

    if isinstance(description, str) and len(description) > MAX_DESCRIPTION_CHARS:
        payload["description"] = description[:MAX_DESCRIPTION_CHARS]

    return payload, None
