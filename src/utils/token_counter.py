from __future__ import annotations

import tiktoken

try:
    _enc = tiktoken.encoding_for_model("text-embedding-3-large")
except KeyError:
    _enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text*."""
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return *text* truncated to at most *max_tokens* tokens."""
    return _enc.decode(_enc.encode(text)[:max_tokens])
