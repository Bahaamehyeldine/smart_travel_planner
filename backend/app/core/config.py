from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent.parent

class Settings(BaseSettings):
    DATABASE_URL: str
    GROQ_API_KEY: str | None = None

    # Ignore unrelated keys so an older .env keeps working.
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
