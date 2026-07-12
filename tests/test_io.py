from __future__ import annotations

from pathlib import Path

import numpy as np

from rmv_watermarks.core.io import decode_bytes, encode_bytes, imread, imwrite, to_bgr


def test_to_bgr_from_gray() -> None:
  gray = np.zeros((10, 10), dtype=np.uint8)
  bgr = to_bgr(gray)
  assert bgr.shape == (10, 10, 3)


def test_to_bgr_from_bgra() -> None:
  bgra = np.zeros((10, 10, 4), dtype=np.uint8)
  bgr = to_bgr(bgra)
  assert bgr.shape == (10, 10, 3)


def test_imwrite_imread_roundtrip(tmp_path: Path) -> None:
  image = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
  path = tmp_path / "out.png"
  assert imwrite(path, image)
  loaded = imread(path)
  assert loaded is not None
  assert loaded.shape == image.shape


def test_bytes_roundtrip() -> None:
  image = np.full((16, 16, 3), 42, dtype=np.uint8)
  data = encode_bytes(image, ".png")
  decoded = decode_bytes(data)
  assert decoded is not None
  assert decoded.shape == image.shape
