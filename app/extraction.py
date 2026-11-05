"""Толерантное извлечение JSON из ответа модели.

Модель обязана отвечать чистым JSON (контракт промпта), но на практике
ответ приходит в markdown-заборе или с преамбулой — парсим устойчиво:
сначала прямая попытка, затем первый {...} в тексте (паттерн ExtractJson
из LlmOneShot боевой платформы). Гарантия корректности не нужна —
структуру ответа дополнительно проверяет генератор.
"""

import json
import re

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text: str) -> dict | None:
    """Первый JSON-объект из ответа модели. None — JSON не найден."""
    if not text:
        return None
    stripped = text.strip()
    #markdown-забор: ```json ... ``` или ``` ... ```
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```").strip()
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_OBJECT_RE.search(stripped)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
