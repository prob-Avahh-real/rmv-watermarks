from __future__ import annotations

import numpy as np


def alpha_from_capture(bg_capture: np.ndarray) -> np.ndarray:
  """Derive a normalized alpha map from a watermark-on-solid-background capture."""
  if bg_capture.ndim == 3 and bg_capture.shape[2] >= 3:
    gray = np.max(bg_capture[..., :3], axis=2)
  else:
    gray = bg_capture
  return gray.astype(np.float32) / 255.0


def apply_watermark(
  region: np.ndarray,
  alpha: np.ndarray,
  *,
  logo_value: float = 255.0,
) -> np.ndarray:
  """Forward alpha blend: watermarked = alpha * logo + (1 - alpha) * original."""
  image_f = region.astype(np.float32)
  alpha_3ch = np.expand_dims(alpha, axis=2)
  blended = alpha_3ch * logo_value + (1.0 - alpha_3ch) * image_f
  return np.clip(blended, 0, 255).astype(np.uint8)


def reverse_alpha_blend(
  region: np.ndarray,
  alpha: np.ndarray,
  *,
  logo_value: float = 255.0,
  alpha_threshold: float = 0.002,
  max_alpha: float = 0.99,
) -> np.ndarray:
  """Recover original pixels: original = (watermarked - alpha * logo) / (1 - alpha)."""
  image_f = region.astype(np.float32)
  alpha_3ch = np.expand_dims(alpha, axis=2)
  valid = alpha >= alpha_threshold
  alpha_clamped = np.minimum(alpha_3ch, max_alpha)
  one_minus_alpha = 1.0 - alpha_clamped
  restored = (image_f - alpha_clamped * logo_value) / one_minus_alpha
  result = image_f.copy()
  for channel in range(3):
    result[:, :, channel] = np.where(valid, restored[:, :, channel], image_f[:, :, channel])
  return np.clip(result, 0, 255).astype(np.uint8)
