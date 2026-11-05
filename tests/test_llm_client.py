"""Клиент LLM против мок-транспорта (httpx2.MockTransport — SDK openai 3.x
работает поверх httpx2, поэтому respx/httpx тут не перехватывают).

Ретраи 429/5xx выполняет OpenAI SDK на своём уровне; тест проверяет итог:
сколько раз ушло в сеть и как классифицирован финальный сбой. Ключей и сети не нужно.
"""

import httpx2

from app.config import Settings
from app.llm import LlmClient, LlmError

BASE = "http://test.local/v1"


def ok_body(content: str = "текст ответа") -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class MockApi:
    """Очередь ответов/исключений на каждый HTTP-вызов; помнит запросы."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if not self.outcomes:
            return httpx2.Response(500, json={"error": {"message": "queue exhausted"}})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def calls(self) -> int:
        return len(self.requests)


def make_client(mock: MockApi, max_retries: int = 2) -> LlmClient:
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url=BASE,
        llm_model="test-model",
        llm_timeout_s=5,
        llm_max_retries=max_retries,
    )
    http = httpx2.Client(transport=httpx2.MockTransport(mock), base_url=BASE)
    return LlmClient(settings, http_client=http)


def test_retry_on_429_then_success():
    """429 с Retry-After → SDK ретраит → второй ответ принимается."""
    mock = MockApi(
        httpx2.Response(429, headers={"Retry-After": "0"}, json={"error": {"message": "rate limited"}}),
        httpx2.Response(200, json=ok_body()),
    )
    assert make_client(mock).chat("sys", "user") == "текст ответа"
    assert mock.calls == 2


def test_retry_on_500_then_success():
    mock = MockApi(
        httpx2.Response(500, json={"error": {"message": "boom"}}),
        httpx2.Response(200, json=ok_body()),
    )
    assert make_client(mock).chat("sys", "user") == "текст ответа"
    assert mock.calls == 2


def test_401_no_retry_classified():
    mock = MockApi(httpx2.Response(401, json={"error": {"message": "bad key"}}))
    client = make_client(mock)
    try:
        client.chat("sys", "user")
        raise AssertionError("ожидалась LlmError")
    except LlmError as e:
        assert e.status == 502
        assert "LLM_API_KEY" in str(e)
    assert mock.calls == 1  # 4xx без ретрая


def test_429_exhausted():
    mock = MockApi(*[httpx2.Response(429, headers={"Retry-After": "0"}, json={})] * 5)
    client = make_client(mock, max_retries=2)
    try:
        client.chat("sys", "user")
        raise AssertionError("ожидалась LlmError")
    except LlmError as e:
        assert "лимит" in str(e).lower()
    assert mock.calls == 3  # 1 попытка + 2 ретрая


def test_connection_error():
    mock = MockApi(httpx2.ConnectError("refused"))
    client = make_client(mock, max_retries=0)
    try:
        client.chat("sys", "user")
        raise AssertionError("ожидалась LlmError")
    except LlmError as e:
        assert e.status == 502
        assert "недоступен" in str(e)


def test_timeout_classified():
    mock = MockApi(httpx2.ReadTimeout("slow"))
    client = make_client(mock, max_retries=0)
    try:
        client.chat("sys", "user")
        raise AssertionError("ожидалась LlmError")
    except LlmError as e:
        assert e.status == 502


def test_empty_content_raises():
    """Лимит токенов съеден размышлениями модели — content пуст, это ошибка."""
    mock = MockApi(httpx2.Response(200, json=ok_body(None)))
    client = make_client(mock)
    try:
        client.chat("sys", "user")
        raise AssertionError("ожидалась LlmError")
    except LlmError as e:
        assert "пустой ответ" in str(e)


def test_request_shape():
    """Запрос уходит chat.completions с system+user и temperature=0."""
    mock = MockApi(httpx2.Response(200, json=ok_body()))
    make_client(mock).chat("SYSTEM", "USER")
    request = mock.requests[0]
    assert request.url.path.endswith("/chat/completions")
    assert request.url.host == "test.local"
    import json
    body = json.loads(request.content)
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    assert body["temperature"] == 0.0
