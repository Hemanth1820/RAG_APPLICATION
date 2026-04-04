from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AWS
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")

    # Bedrock
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        alias="BEDROCK_MODEL_ID",
    )
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v1",
        alias="BEDROCK_EMBED_MODEL_ID",
    )

    # LangGraph checkpointer
    checkpoint_db_path: str = Field(
        default="./data/checkpoints.db",
        alias="CHECKPOINT_DB_PATH",
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection_name: str = Field(
        default="mara_docs",
        alias="CHROMA_COLLECTION_NAME",
    )

    # Document processing
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_retrieval_docs: int = Field(default=4, alias="MAX_RETRIEVAL_DOCS")

    # Server
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")


settings = Settings()
