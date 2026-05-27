from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chat_model: str = "gemini-2.5-flash-lite"
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection: str = "about_me"

    cors_allow_origins: str = ""

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        if not self.cors_allow_origins:
            return []
        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]

    class Config:
        env_file = ".env"


settings = Settings()