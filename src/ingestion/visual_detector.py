from __future__ import annotations

from src.config.settings import settings


def is_visual_page(
    page_text: str,
    image_count: int,
    text_area_fraction: float,
) -> bool:
    """Return True if this page likely contains visual content.

    Conditions (any one is sufficient):
      1. At least one embedded image.
      2. Text area fraction is below the configured visual content threshold.
      3. The text contains structural table markers: 3+ lines each with
         2+ pipe characters, OR 3+ lines each with 2+ tab characters.
    """
    # Condition 1: embedded image AND low text coverage.
    # Requiring both prevents a per-page logo (common in Block Scholes PDFs)
    # from flagging every page as visual when the page is otherwise text-dense.
    if image_count >= 1 and text_area_fraction < settings.VISUAL_CONTENT_THRESHOLD:
        return True

    # Condition 2: sparse text coverage
    if text_area_fraction < settings.VISUAL_CONTENT_THRESHOLD:
        return True

    # Condition 3: structural table markers
    lines = page_text.splitlines()
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    tab_lines = sum(1 for line in lines if line.count("\t") >= 2)
    if pipe_lines >= 3 or tab_lines >= 3:
        return True

    return False
