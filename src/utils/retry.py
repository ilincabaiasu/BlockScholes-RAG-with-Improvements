from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_MAX_ATTEMPTS = 4
_BASE_WAIT_S = 1.0

T = TypeVar("T")


async def with_exponential_backoff(
    fn: Callable[[], Awaitable[T]],
    retry_on: type[BaseException] | tuple[type[BaseException], ...],
    label: str,
) -> T:
    """Call *fn* and retry on *retry_on* exceptions with exponential backoff.

    Makes up to 4 attempts (1 original + 3 retries). Wait starts at 1 s
    and doubles each time. Raises on the 4th consecutive failure.

    Args:
        fn:       Zero-argument async callable to execute.
        retry_on: Exception type(s) that trigger a retry.
        label:    Identifier logged with each retry warning (e.g. "openai.embed").
    """
    attempt = 0
    wait = _BASE_WAIT_S

    while True:
        try:
            return await fn()
        except retry_on:
            attempt += 1
            if attempt >= _MAX_ATTEMPTS:
                raise
            _logger.warning(
                "rate_limit_retry",
                extra={"label": label, "attempt": attempt, "wait_s": wait},
            )
            await asyncio.sleep(wait)
            wait *= 2
