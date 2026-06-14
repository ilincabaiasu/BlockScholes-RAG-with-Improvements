from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API credentials (must be set in .env)
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    COHERE_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str

    # Model names
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_VISION_MODEL: str = "gemini-2.5-flash"
    OPENAI_MODEL: str = "gpt-4o"
    RERANKER_MODEL: str = "rerank-english-v3.0"
    GENERATION_PROVIDER: str = "gemini"  # "gemini" | "openai"
    # Ablation LLM judge. "openai" (gpt-4o) is the unbiased default — a different
    # model from GENERATION_PROVIDER. "gemini" reuses the Gemini key but, when it
    # matches the generation model, introduces self-grading bias (scores skew
    # high/relative). The judge logs a warning whenever it self-grades.
    JUDGE_PROVIDER: str = "openai"  # "openai" | "gemini"

    # Retrieval
    DENSE_TOP_K: int = 30
    SPARSE_TOP_K: int = 30
    RERANKER_TOP_K: int = 20
    CONTEXT_TOP_K: int = 10
    MAX_CHUNKS_PER_DOC: int = 2  # diversity cap: max chunks from one source
    RERANKER_MIN_SCORE: float = 0.0
    RRF_K: int = 60

    # Chunking
    FIXED_CHUNK_TOKENS: int = 500
    FIXED_CHUNK_OVERLAP: int = 50
    CHILD_CHUNK_TOKENS: int = 300
    PARENT_CHUNK_TOKENS: int = 1500

    # Generation
    MAX_CONTEXT_TOKENS: int = 12000
    MAX_COMPLETION_TOKENS: int = 2048
    TEMP_FACTUAL: float = 0.0
    TEMP_ANALYTICAL: float = 0.2
    CRITIQUE_CONFIDENCE_THRESHOLD: float = 0.7

    # Visual retrieval
    VISUAL_CONTENT_THRESHOLD: float = 0.3
    PAGE_RENDER_DPI: int = 150

    # Qdrant
    QDRANT_COLLECTION: str = "block_scholes"
    EMBEDDING_DIM: int = 3072  # gemini-embedding-001 output size


settings = Settings()
