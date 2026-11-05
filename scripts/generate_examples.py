"""Генерация примеров результата для README: data/products/*.json → examples/*.json.

Запуск (нужен .env с ключом):
    python scripts/generate_examples.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.generator import generate_card  # noqa: E402
from app.llm import LlmClient  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.llm_api_key:
        print("LLM_API_KEY не задан: скопируйте .env.example в .env и укажите ключ")
        return 1

    llm = LlmClient(settings)
    out_dir = ROOT / "examples"
    out_dir.mkdir(exist_ok=True)

    for path in sorted((ROOT / "data" / "products").glob("*.json")):
        product = json.loads(path.read_text(encoding="utf-8"))
        print(f"Генерирую: {path.name} …", flush=True)
        try:
            card = generate_card(product, llm)
        except Exception as e:  # пример не должен ронять весь прогон
            print(f"  ОШИБКА: {e}")
            continue
        out = out_dir / f"result-{path.stem}.json"
        out.write_text(json.dumps(card.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
