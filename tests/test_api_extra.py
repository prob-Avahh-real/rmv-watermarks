from __future__ import annotations

import numpy as np
import pytest

from rmv_watermarks.api import detect, remove
from rmv_watermarks.core.io import imread


def test_detect_ndarray_input(small_watermarked: np.ndarray) -> None:
  result = detect(small_watermarked)
  assert result.detected


def test_remove_ndarray_no_dest(small_watermarked: np.ndarray) -> None:
  result = remove(small_watermarked, force=True)
  assert result.removed
  assert result.image.shape == small_watermarked.shape


def test_remove_auto_provider(small_watermarked: np.ndarray) -> None:
  result = remove(small_watermarked, provider="auto", force=True)
  assert result.removed


def test_imread_missing_file() -> None:
  assert imread("/nonexistent/path/image.png") is None


def test_api_file_not_found() -> None:
  with pytest.raises(FileNotFoundError):
    detect("/does/not/exist.png")


def test_api_invalid_bytes() -> None:
  with pytest.raises(ValueError, match="Failed to decode"):
    remove(b"not-an-image", force=True)
