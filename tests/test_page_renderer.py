from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.utils.page_renderer import PageRenderError, clear_cache, render_page


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the page render cache before every test."""
    clear_cache()
    yield
    clear_cache()


def _make_pdf2image_mock(image):
    """Return a mock pdf2image module whose convert_from_path returns [image]."""
    mock_module = MagicMock()
    mock_module.convert_from_path.return_value = [image]
    return mock_module


def test_render_page_returns_image():
    mock_image = MagicMock()
    mock_pdf2image = _make_pdf2image_mock(mock_image)

    with patch.dict(sys.modules, {"pdf2image": mock_pdf2image}):
        with patch("src.utils.page_renderer.os.path.isfile", return_value=True):
            result = render_page("fake.pdf", 1)

    assert result is mock_image


def test_render_page_uses_cache():
    mock_image = MagicMock()
    mock_pdf2image = _make_pdf2image_mock(mock_image)

    with patch.dict(sys.modules, {"pdf2image": mock_pdf2image}):
        with patch("src.utils.page_renderer.os.path.isfile", return_value=True):
            result1 = render_page("fake.pdf", 1)
            result2 = render_page("fake.pdf", 1)

    assert result1 is result2
    # convert_from_path should have been called exactly once
    assert mock_pdf2image.convert_from_path.call_count == 1


def test_render_page_missing_file_raises():
    with patch("src.utils.page_renderer.os.path.isfile", return_value=False):
        with pytest.raises(PageRenderError):
            render_page("nonexistent.pdf", 1)
