"""Извлечение входов из выгрузки прогона (xlsx) в data/testset/*.json.

Использование: python scripts/extract_testset.py <путь-к-xlsx>
Требует openpyxl (pip install openpyxl). Читает только колонку «Запрос (вход агента)».
"""

import json
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "testset"


def main(xlsx_path: str) -> int:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.worksheets[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        case_id, _run_id, product_name, request_json = row[0], row[1], row[2], row[3]
        if not isinstance(case_id, str) or not case_id.startswith("SP-"):
            continue
        try:
            product = json.loads(request_json)
        except (json.JSONDecodeError, TypeError):
            print(f"Пропуск {case_id}: вход не разобран")
            continue
        out = OUT_DIR / f"{case_id.lower()}.json"
        out.write_text(json.dumps(
            {"id": case_id, "product_name": product_name, "product": product},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count += 1
    print(f"Извлечено кейсов: {count} → {OUT_DIR}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
