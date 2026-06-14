from __future__ import annotations

import json
import os

from src.pipelines.models import QueryResult
from src.utils.logger import get_logger

_logger = get_logger(__name__)

LOG_PATH = "data/query_log.jsonl"


def log_result(result: QueryResult) -> None:
    """Append *result* as a JSON line to LOG_PATH.

    Creates the parent directory if it does not exist.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.model_dump()) + "\n")

    _logger.info(
        "log_result",
        extra={
            "query_id": result.query_id,
            "pipeline": result.pipeline,
            "visual_path_used": result.visual_path_used,
        },
    )


def load_results(pipeline: str | None = None) -> list[dict]:
    """Read all logged QueryResult dicts from LOG_PATH.

    Args:
        pipeline: If given, return only entries where pipeline matches.

    Returns:
        List of dicts. Empty list if LOG_PATH does not exist.
    """
    if not os.path.exists(LOG_PATH):
        return []

    results: list[dict] = []
    with open(LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if pipeline is None or entry.get("pipeline") == pipeline:
                results.append(entry)

    return results
