from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    provider: str = ""       # "gemini" | "gemini-vision" | "openai"
    source_page: int = 0
    source_doc: str = ""
