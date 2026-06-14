from __future__ import annotations

import time
from types import TracebackType
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class Timer:
    """Context manager that measures wall-clock time for a named pipeline stage.

    Usage::

        with Timer("retrieve") as t:
            results = index.search(query)
        print(t.elapsed_ms)  # available after the with-block

    On exit, logs one JSON line::

        {"stage": "<label>", "elapsed_ms": <float>}
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000
        _logger.info(
            "timer",
            extra={"stage": self.label, "elapsed_ms": round(self.elapsed_ms, 3)},
        )
