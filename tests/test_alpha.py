from __future__ import annotations

import numpy as np
import pytest

from rmv_watermarks.core.alpha import apply_watermark, reverse_alpha_blend
from rmv_watermarks.providers.gemini import GeminiProvider
from rmv_watermarks.types import WatermarkSize


def test_round_trip_preserves_pixels() -> None:
  provider = GeminiProvider()
  alpha = provider._alpha_for(WatermarkSize.SMALL)
  rng = np.random.default_rng(0)
  region = rng.integers(0, 256, alpha.shape + (3,), dtype=np.uint8)

  watermarked = apply_watermark(region, alpha)
  restored = reverse_alpha_blend(watermarked, alpha)

  mask = (alpha >= 0.002) & (alpha <= 0.99)
  diff = np.abs(restored.astype(np.int16) - region.astype(np.int16))
  assert diff[mask].max() <= 2


def test_reverse_ignores_low_alpha() -> None:
  region = np.full((8, 8, 3), 100, dtype=np.uint8)
  alpha = np.zeros((8, 8), dtype=np.float32)
  alpha[2:6, 2:6] = 0.5

  watermarked = apply_watermark(region, alpha)
  restored = reverse_alpha_blend(watermarked, alpha)

  assert np.array_equal(restored[:2, :2], region[:2, :2])
  assert not np.array_equal(restored[3:5, 3:5], watermarked[3:5, 3:5])


def test_alpha_from_capture_normalizes() -> None:
  from rmv_watermarks.core.alpha import alpha_from_capture

  capture = np.zeros((4, 4, 3), dtype=np.uint8)
  capture[0, 0] = (0, 0, 0)
  capture[1, 1] = (0, 128, 0)
  capture[2, 2] = (0, 0, 255)
  alpha = alpha_from_capture(capture)
  assert alpha.shape == (4, 4)
  assert alpha.max() == pytest.approx(1.0)
  assert alpha.min() == pytest.approx(0.0)
