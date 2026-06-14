from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# GenerationResult lives in src/generation/models — re-exported here so that
# pipeline modules can still do `from src.pipelines.models import GenerationResult`.
from src.generation.models import GenerationResult

__all__ = ["GenerationResult", "QueryResult"]


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    original_query: str
    rewritten_query: str = ""
    query_type: str = ""     # factual_lookup|analytical|comparative|definitional
    pipeline: str = ""       # "gemini" | "baseline" | "enhanced"
    retrieved_chunk_ids: list[str] = []
    reranker_scores: list[float] = []
    context_sources: list[str] = []
    response_text: str = ""
    refined: bool = False
    citation_verification: dict = {}
    latency_breakdown: dict = {}
    token_usage: dict = {}
    visual_path_used: bool = False
    visual_pages_rendered: list[int] = []
    generation_provider: str = ""
    timestamp: str = ""      # ISO-8601 datetime string
