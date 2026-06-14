from __future__ import annotations

from src.utils.token_counter import count_tokens, truncate_to_tokens


def test_count_tokens_nonempty():
    result = count_tokens("hello world")
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_truncate_to_tokens_shortens_text():
    original = "hello world hello"
    truncated = truncate_to_tokens(original, max_tokens=2)
    assert len(truncated) < len(original)
