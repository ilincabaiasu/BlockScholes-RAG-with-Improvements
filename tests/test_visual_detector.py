from __future__ import annotations

from src.ingestion.visual_detector import is_visual_page


def test_pure_prose_returns_false():
    text = "This is a plain prose paragraph with no tables or images."
    assert is_visual_page(text, image_count=0, text_area_fraction=0.8) is False


def test_image_present_returns_true():
    assert is_visual_page("any text", image_count=1, text_area_fraction=0.5) is True


def test_low_fraction_returns_true():
    # fraction=0.1 < VISUAL_CONTENT_THRESHOLD (0.3) → True
    assert is_visual_page("any text", image_count=0, text_area_fraction=0.1) is True


def test_pipe_table_returns_true():
    # 4 lines each with 2+ pipes
    lines = ["col1 | col2 | col3"] * 4
    text = "\n".join(lines)
    assert is_visual_page(text, image_count=0, text_area_fraction=0.6) is True


def test_tab_table_returns_true():
    # 4 lines each with 2+ tabs
    lines = ["col1\tcol2\tcol3"] * 4
    text = "\n".join(lines)
    assert is_visual_page(text, image_count=0, text_area_fraction=0.6) is True
