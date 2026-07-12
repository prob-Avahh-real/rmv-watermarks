from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rmv_watermarks.api import detect, list_providers, remove


def test_list_providers() -> None:
  assert "gemini" in list_providers()


def test_detect_on_watermarked(small_watermarked: np.ndarray) -> None:
  result = detect(small_watermarked)
  assert result.detected


def test_remove_file_api(tmp_path: Path, small_watermarked: np.ndarray) -> None:
  src = tmp_path / "in.png"
  dst = tmp_path / "out.png"
  import cv2

  cv2.imwrite(str(src), small_watermarked)
  result = remove(src, dst, force=True)
  assert result.removed
  assert dst.exists()


def test_remove_bytes_api(small_watermarked: np.ndarray) -> None:
  from rmv_watermarks.core.io import encode_bytes

  data = encode_bytes(small_watermarked)
  result, out_bytes = remove(data, return_bytes=True, force=True)
  assert result.removed
  assert len(out_bytes) > 0


def test_remove_auto_skips_clean(gradient_image: np.ndarray) -> None:
  result = remove(gradient_image, provider="gemini")
  assert not result.removed


def test_remove_unknown_provider(gradient_image: np.ndarray) -> None:
  with pytest.raises(KeyError):
    remove(gradient_image, provider="missing")
