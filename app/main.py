"""FastAPI-приложение: веб-форма + JSON API. Запуск: uvicorn app.main:app."""

import json
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .generator import SemanticError, generate_card
from .llm import LlmClient, LlmError
from .validation import parse_and_validate

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_DIR = ROOT / "data" / "products"

app = FastAPI(title="AI-помощник для подготовки товарных описаний (КИП)")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

_settings = get_settings()
_llm: LlmClient | None = LlmClient(_settings) if _settings.llm_api_key else None


def get_llm() -> LlmClient:
    """DI-точка: тесты подменяют через dependency_overrides."""
    if _llm is None:
        raise LlmError("LLM_API_KEY не задан — скопируйте .env.example в .env и укажите ключ", 503)
    return _llm


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": _llm is not None, "model": _settings.llm_model}


@app.get("/api/examples")
def examples():
    """Два товара из задания — для кнопок в веб-форме."""
    items = []
    for path in sorted(PRODUCTS_DIR.glob("*.json")):
        items.append({"file": path.name, "product": json.loads(path.read_text(encoding="utf-8"))})
    return {"items": items}


@app.exception_handler(LlmError)
async def llm_error_handler(_request: Request, exc: LlmError):
    """Ошибки LLM-слоя (включая поднятие из Depends(get_llm)) → JSON, не 500."""
    return JSONResponse({"error": str(exc)}, status_code=exc.status)


@app.exception_handler(SemanticError)
async def semantic_error_handler(_request: Request, exc: SemanticError):
    return JSONResponse({"error": exc.message}, status_code=422)


@app.post("/api/card")
async def create_card(request: Request, llm: LlmClient = Depends(get_llm)):
    raw = (await request.body()).decode("utf-8", errors="replace")

    product, error = parse_and_validate(raw)
    if error:
        return JSONResponse({"error": error}, status_code=400)

    card = generate_card(product, llm)
    return JSONResponse(card.model_dump())
