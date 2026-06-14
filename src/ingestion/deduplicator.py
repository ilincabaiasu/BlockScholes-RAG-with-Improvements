from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_HASH_FILE: Path = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "doc_hashes.json"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_hashes() -> dict[str, str]:
    """Return the persisted doc_id → sha256_hash mapping.

    Returns an empty dict if the file does not exist or is corrupted.
    """
    if not _HASH_FILE.is_file():
        return {}
    try:
        return json.loads(_HASH_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _logger.warning(
            "hash_file_unreadable",
            extra={"path": str(_HASH_FILE), "error": str(exc)},
        )
        return {}


def is_duplicate(clean_text: str, existing_hashes: dict) -> bool:
    """Return True if the sha256 of *clean_text* already exists in *existing_hashes*."""
    return _sha256(clean_text) in existing_hashes.values()


def save_hash(doc_id: str, clean_text: str, existing_hashes: dict) -> dict:
    """Add *doc_id* → sha256 to *existing_hashes*, persist to disk, return updated dict.

    Writes atomically via a temp file + rename so a crash cannot corrupt the store.
    Creates data/processed/ if it does not exist.
    """
    updated = {**existing_hashes, doc_id: _sha256(clean_text)}
    _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write to a sibling temp file, then rename — atomic on POSIX
    tmp_fd, tmp_path = tempfile.mkstemp(dir=_HASH_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
        os.replace(tmp_path, _HASH_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise

    return updated
