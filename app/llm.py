"""OpenAI-совместимый клиент одной LLM. Эндпоинт/модель/ключ — из .env, работает
с OpenAI, OpenRouter, Z.ai, локальными моделями (Ollama/llama.cpp).

Ретраи 429/5xx/сетевых ошибок с экспоненциальным backoff и учётом Retry-After
делает сам OpenAI SDK (max_retries). Сюда попадает только итог: либо ответ,
либо классифицированная ошибка без ретрая (4xx аутентификация/запрос).
"""

from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from .config import Settings


class LlmError(Exception):
    """Ошибка вызова LLM API после всех ретраев. status — HTTP-код для ответа сервиса."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class LlmClient:
    def __init__(self, settings: Settings, http_client=None):
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        client_kwargs = dict(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        # http_client — точка подмены транспорта в тестах (httpx2.MockTransport)
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = OpenAI(**client_kwargs)

    def chat(self, system: str, user: str) -> str:
        """Одиночный вызов без инструментов и истории. Возвращает текст ответа."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except AuthenticationError:
            raise LlmError("Ошибка аутентификации LLM API — проверьте LLM_API_KEY", 502)
        except NotFoundError:
            raise LlmError("LLM API не нашёл модель — проверьте LLM_MODEL и LLM_BASE_URL", 502)
        except BadRequestError as e:
            raise LlmError(f"LLM API отклонил запрос: {e.message}", 502)
        except RateLimitError:
            # SDK уже ретраил с backoff и Retry-After — сюда попадаем только по исчерпании
            raise LlmError("Превышен лимит запросов LLM API — повторите позже", 502)
        except APIConnectionError as e:
            # APITimeoutError — подкласс: таймаут и обрыв соединения обрабатываются одинаково
            raise LlmError(f"LLM API недоступен: {e.__class__.__name__}", 502)
        except InternalServerError:
            raise LlmError("LLM API отвечает ошибкой 5xx после всех повторов", 502)
        except APIStatusError as e:
            raise LlmError(f"Неожиданный ответ LLM API (HTTP {e.status_code})", 502)

        content = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not content:
            # Пустой ответ чаще всего = лимит токенов съеден размышлениями модели
            raise LlmError("LLM вернул пустой ответ — проверьте LLM_MAX_TOKENS и настройки модели", 502)
        return content
