from __future__ import annotations

import numpy as np

from rmv_watermarks.providers.gemini import GeminiProvider
from rmv_watermarks.types import WatermarkSize


def test_detects_synthetic_watermark(
  provider: GeminiProvider, small_watermarked: np.ndarray
) -> None:
  result = provider.detect(small_watermarked)
  assert result.detected
  assert result.confidence >= 0.35
  assert result.region is not None


def test_no_detection_on_clean_image(provider: GeminiProvider, gradient_image: np.ndarray) -> None:
  result = provider.detect(gradient_image)
  assert not result.detected


def test_remove_restores_original(
  provider: GeminiProvider,
  gradient_image: np.ndarray,
  small_watermarked: np.ndarray,
) -> None:
  removal = provider.remove(small_watermarked, force=True)
  assert removal.removed
  diff = np.abs(removal.image.astype(np.int16) - gradient_image.astype(np.int16))
  assert diff.max() <= 2


def test_large_watermark_roundtrip(provider: GeminiProvider, large_image: np.ndarray) -> None:
  marked = provider.add_watermark(large_image)
  removal = provider.remove(marked, force=True)
  assert removal.removed
  diff = np.abs(removal.image.astype(np.int16) - large_image.astype(np.int16))
  assert diff.max() <= 2


def test_skip_when_not_detected(provider: GeminiProvider, gradient_image: np.ndarray) -> None:
  removal = provider.remove(gradient_image)
  assert not removal.removed
  assert removal.skipped_reason == "no_watermark_detected"


def test_force_removes_without_detection(
  provider: GeminiProvider, gradient_image: np.ndarray
) -> None:
  removal = provider.remove(gradient_image, force=True)
  assert removal.removed


def test_force_size_override(provider: GeminiProvider, gradient_image: np.ndarray) -> None:
  marked = provider.add_watermark(gradient_image, force_size=WatermarkSize.LARGE)
  detection = provider.detect(marked, force_size=WatermarkSize.LARGE)
  assert detection.detected
  assert detection.size == WatermarkSize.LARGE


def test_grayscale_input(provider: GeminiProvider, small_watermarked: np.ndarray) -> None:
  gray = np.mean(small_watermarked, axis=2).astype(np.uint8)
  removal = provider.remove(gray, force=True)
  assert removal.removed
  assert removal.image.ndim == 3


def test_rgba_input(provider: GeminiProvider, small_watermarked: np.ndarray) -> None:
  rgba = np.dstack([small_watermarked, np.full(small_watermarked.shape[:2], 255, dtype=np.uint8)])
  removal = provider.remove(rgba, force=True)
  assert removal.removed
