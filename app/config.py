"""Настройки сервиса из переменных окружения / .env. Ключ API в коде не хранится."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16000


@lru_cache
def get_settings() -> Settings:
    return Settings()
