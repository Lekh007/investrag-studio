from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings with safe local defaults."""

    model_config = SettingsConfigDict(env_prefix="INVESTRAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".data/investrag")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "granite4.2:3b"
    ollama_disable_thinking: bool = True
    ollama_temperature: float = 0.1
    embedding_model: str = "BAAI/bge-m3"
    embedding_mode: str = "auto"
    embedding_fallback_dimension: int = 512
    max_upload_bytes: int = 100 * 1024 * 1024
    cors_origins: str = "http://localhost:5174,http://127.0.0.1:5174"
    publish_vector_stores: str = "chroma,qdrant"
    max_context_tokens: int = 8000
    postgres_dsn: str | None = None
    catalog_dsn: str | None = None
    weaviate_host: str | None = None
    weaviate_port: int = 8090
    weaviate_grpc_port: int = 50052
    pinecone_api_key: str | None = None
    pinecone_index: str = "investrag"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.db"

    @property
    def allowed_origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]

    @property
    def publication_stores(self) -> list[str]:
        return [value.strip() for value in self.publish_vector_stores.split(",") if value.strip() and value.strip() != "faiss"]
