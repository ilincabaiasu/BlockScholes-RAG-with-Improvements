from __future__ import annotations

import numpy as np
from langchain_text_splitters import TokenTextSplitter
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.utils.token_counter import count_tokens

_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Fallback splitter for individual paragraphs that still exceed the ceiling
# after paragraph-boundary splitting — no overlap so chunks are independent.
# encoding_name must match count_tokens (cl100k_base) so token budgets agree.
_fallback_splitter = TokenTextSplitter(
    chunk_size=settings.CHILD_CHUNK_TOKENS,
    chunk_overlap=0,
    encoding_name="cl100k_base",
)

_SIMILARITY_THRESHOLD = 0.5
_MIN_PARA_CHARS = 20


def chunk_semantic(text: str) -> list[str]:
    """Split *text* into semantically coherent segments.

    Returns a list of text strings. Called by the hierarchical chunker.
    """
    # 1. Split into paragraphs and filter short ones
    paragraphs = [p for p in text.split("\n\n") if len(p.strip()) >= _MIN_PARA_CHARS]

    # 2. Degenerate case — still enforce ceiling
    if len(paragraphs) < 2:
        if count_tokens(text) > settings.CHILD_CHUNK_TOKENS:
            return _fallback_splitter.split_text(text)
        return [text]

    # 3. Embed all paragraphs in one batch
    embeddings = _embed_model.encode(paragraphs, convert_to_numpy=True)

    # 4. L2-normalise then cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    normed = embeddings / norms
    similarities = (normed[:-1] * normed[1:]).sum(axis=1)  # shape: (n-1,)

    # 5. Detect split points where similarity drops below threshold
    split_after: set[int] = {
        i for i, sim in enumerate(similarities) if sim < _SIMILARITY_THRESHOLD
    }

    # 6. Merge consecutive paragraphs between split points into segments
    segments: list[str] = []
    current: list[str] = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        if (i - 1) in split_after:
            segments.append("\n\n".join(current))
            current = [paragraphs[i]]
        else:
            current.append(paragraphs[i])

    segments.append("\n\n".join(current))

    # 7. Enforce hard token ceiling by re-splitting at paragraph boundaries
    ceiling = settings.CHILD_CHUNK_TOKENS
    final: list[str] = []

    for segment in segments:
        if count_tokens(segment) <= ceiling:
            final.append(segment)
            continue

        # Re-split this segment at paragraph boundaries
        paras = segment.split("\n\n")
        bucket: list[str] = []
        bucket_tokens = 0

        for para in paras:
            para_tokens = count_tokens(para)
            if bucket and bucket_tokens + para_tokens > ceiling:
                final.append("\n\n".join(bucket))
                bucket = [para]
                bucket_tokens = para_tokens
            else:
                bucket.append(para)
                bucket_tokens += para_tokens

        if bucket:
            final.append("\n\n".join(bucket))

    # Fallback: if any item still exceeds the ceiling (single paragraph with no
    # \n\n boundaries), split it with TokenTextSplitter as a last resort.
    enforced: list[str] = []
    for item in final:
        if count_tokens(item) > ceiling:
            enforced.extend(_fallback_splitter.split_text(item))
        else:
            enforced.append(item)
    final = enforced

    return final
