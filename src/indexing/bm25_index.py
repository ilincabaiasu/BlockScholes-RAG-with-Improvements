from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger

_logger = get_logger(__name__)

PICKLE_PATH = Path("data/processed/bm25_index.pkl")
IDS_PATH = Path("data/processed/bm25_chunk_ids.json")

# Split on whitespace and punctuation, but preserve hyphens between
# alphanumeric chars (e.g. "25-delta", "vol-of-vol", "at-the-money").
_SPLIT_RE = re.compile(r"(?<![A-Za-z0-9])-|-(?![A-Za-z0-9])|[\s,;:!?()[\]{}<>\"'`~/\\|@#$%^&*+=]+")


def tokenise(text: str) -> list[str]:
    """Tokenise *text* for BM25, preserving hyphenated financial terms."""
    tokens = _SPLIT_RE.split(text.lower())
    return [t for t in tokens if t]


def build_index(chunks: list) -> None:
    """Build and persist a BM25Okapi index from child Chunk objects."""
    t0 = time.perf_counter()

    token_lists = [tokenise(c.text) for c in chunks]
    index = BM25Okapi(token_lists)
    chunk_ids = [c.chunk_id for c in chunks]

    PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PICKLE_PATH.open("wb") as f:
        pickle.dump(index, f)

    IDS_PATH.write_text(json.dumps(chunk_ids, indent=2), encoding="utf-8")

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    _logger.info(
        "bm25_index_built",
        extra={"chunk_count": len(chunks), "elapsed_ms": round(elapsed_ms, 3)},
    )


def load_index() -> tuple[BM25Okapi | None, list[str]]:
    """Load the BM25 index and chunk ID list from disk.

    Returns (None, []) if either file is missing.
    """
    if not PICKLE_PATH.is_file() or not IDS_PATH.is_file():
        return None, []

    with PICKLE_PATH.open("rb") as f:
        index = pickle.load(f)

    chunk_ids: list[str] = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    return index, chunk_ids


def search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Return top_k (chunk_id, score) tuples for *query* using BM25."""
    index, chunk_ids = load_index()
    if index is None:
        return []

    tokens = tokenise(query)
    scores = index.get_scores(tokens)

    ranked = sorted(
        zip(chunk_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]
