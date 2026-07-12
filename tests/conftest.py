from __future__ import annotations

import numpy as np
import pytest

from rmv_watermarks.providers.gemini import GeminiProvider


@pytest.fixture
def provider() -> GeminiProvider:
  return GeminiProvider()


@pytest.fixture
def gradient_image() -> np.ndarray:
  h, w = 800, 600
  x = np.linspace(0, 255, w, dtype=np.float32)
  y = np.linspace(0, 255, h, dtype=np.float32)
  xv, yv = np.meshgrid(x, y)
  b = xv.astype(np.uint8)
  g = yv.astype(np.uint8)
  r = ((xv + yv) / 2).astype(np.uint8)
  return np.stack([b, g, r], axis=-1)


@pytest.fixture
def small_watermarked(provider: GeminiProvider, gradient_image: np.ndarray) -> np.ndarray:
  return provider.add_watermark(gradient_image)


@pytest.fixture
def large_image() -> np.ndarray:
  h, w = 1200, 1600
  noise = np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)
  return noise


@pytest.fixture
def large_watermarked(provider: GeminiProvider, large_image: np.ndarray) -> np.ndarray:
  return provider.add_watermark(large_image)
