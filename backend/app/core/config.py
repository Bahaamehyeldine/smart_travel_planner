from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# This finds the .env file relative to this file's location
ROOT_DIR = Path(__file__).parent.parent.parent.parent


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ANTHROPIC_API_KEY: str
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    WEATHER_API_KEY: str | None = None
    FLIGHT_API_KEY: str | None = None
    DISCORD_WEBHOOK_URL: str | None = None
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "smart-travel-planner"

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()