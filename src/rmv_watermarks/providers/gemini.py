from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rmv_watermarks.core.alpha import alpha_from_capture, apply_watermark, reverse_alpha_blend
from rmv_watermarks.core.geometry import LAYOUTS, watermark_size_for
from rmv_watermarks.core.io import to_bgr
from rmv_watermarks.protocol import WatermarkProvider
from rmv_watermarks.types import DetectionResult, RemovalResult, WatermarkRegion, WatermarkSize

_DETECT_THRESHOLD = 0.35
_NCC_MIN = 0.25


def _asset_path(name: str) -> Path:
  return Path(__file__).resolve().parent.parent / "assets" / name


def _load_alpha_maps() -> dict[WatermarkSize, np.ndarray]:
  maps: dict[WatermarkSize, np.ndarray] = {}
  for size, filename in (
    (WatermarkSize.SMALL, "bg_48.png"),
    (WatermarkSize.LARGE, "bg_96.png"),
  ):
    path = _asset_path(filename)
    if not path.exists():
      raise FileNotFoundError(f"Missing embedded asset: {path}")
    capture = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if capture is None:
      raise RuntimeError(f"Failed to decode asset: {path}")
    maps[size] = alpha_from_capture(capture)
  return maps


class GeminiProvider(WatermarkProvider):
  """Google Gemini / Nano Banana bottom-right sparkle watermark."""

  name = "gemini"

  def __init__(self, *, logo_value: float = 255.0) -> None:
    self.logo_value = logo_value
    self._alpha_maps = _load_alpha_maps()
    self._alpha_large = self._alpha_maps[WatermarkSize.LARGE]

  def _alpha_for(self, size: WatermarkSize) -> np.ndarray:
    return self._alpha_maps[size]

  def _scaled_alpha(self, scale: int) -> np.ndarray:
    if scale == self._alpha_large.shape[0]:
      return self._alpha_large
    interp = cv2.INTER_LINEAR if scale > self._alpha_large.shape[0] else cv2.INTER_AREA
    return cv2.resize(self._alpha_large, (scale, scale), interpolation=interp)

  def _scan_scales(self, gray_region: np.ndarray) -> list[tuple[float, int, tuple[int, int]]]:
    results: list[tuple[float, int, tuple[int, int]]] = []
    gray_f = gray_region.astype(np.float32) / 255.0
    for scale in range(16, 121, 2):
      if scale > gray_f.shape[0] or scale > gray_f.shape[1]:
        continue
      template = self._scaled_alpha(scale)
      match = cv2.matchTemplate(gray_f, template, cv2.TM_CCOEFF_NORMED)
      _, max_val, _, max_loc = cv2.minMaxLoc(match)
      results.append((float(max_val), scale, max_loc))
    results.sort(reverse=True)
    return results

  def _catalog_region(self, width: int, height: int, size: WatermarkSize) -> WatermarkRegion:
    return LAYOUTS[size].region(width, height)

  def detect(
    self,
    image: np.ndarray,
    *,
    force_size: WatermarkSize | None = None,
  ) -> DetectionResult:
    image = to_bgr(image)
    h, w = image.shape[:2]
    if h == 0 or w == 0:
      return DetectionResult(detected=False, confidence=0.0, provider=self.name)

    size = force_size or watermark_size_for(w, h)

    if force_size is not None:
      region = self._catalog_region(w, h, force_size)
      alpha = self._alpha_for(force_size)
      roi = image[region.y : region.y + region.height, region.x : region.x + region.width]
      gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
      score = float(cv2.matchTemplate(gray, alpha, cv2.TM_CCOEFF_NORMED).max())
      detected = score >= _NCC_MIN
      return DetectionResult(
        detected=detected,
        confidence=min(1.0, max(0.0, score)),
        provider=self.name,
        region=region if detected else None,
        size=force_size,
        details={"ncc": score},
      )

    search = int(min(min(w, h), 512))
    sx, sy = max(0, w - search), max(0, h - search)
    corner = image[sy:h, sx:w]
    gray_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    candidates = self._scan_scales(gray_corner)
    if not candidates:
      return DetectionResult(detected=False, confidence=0.0, provider=self.name)

    best_ncc, best_scale, best_loc = candidates[0]
    detected_region = WatermarkRegion(
      x=sx + best_loc[0],
      y=sy + best_loc[1],
      width=best_scale,
      height=best_scale,
    )

    expected = self._catalog_region(w, h, size)
    ey, ex, eh, ew = expected.y, expected.x, expected.height, expected.width
    default_roi = image[ey : ey + eh, ex : ex + ew]
    default_gray = cv2.cvtColor(default_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    default_alpha = self._alpha_for(size)
    default_ncc = float(cv2.matchTemplate(default_gray, default_alpha, cv2.TM_CCOEFF_NORMED).max())
    if default_ncc >= best_ncc * 0.9:
      detected_region = expected
      best_ncc = max(best_ncc, default_ncc)

    confidence = float(min(1.0, max(0.0, best_ncc)))
    detected = confidence >= _DETECT_THRESHOLD and best_ncc >= _NCC_MIN
    return DetectionResult(
      detected=detected,
      confidence=confidence,
      provider=self.name,
      region=detected_region if detected else None,
      size=size,
      details={"ncc": best_ncc, "scale": best_scale},
    )

  def remove(
    self,
    image: np.ndarray,
    *,
    force: bool = False,
    force_size: WatermarkSize | None = None,
    min_confidence: float = _DETECT_THRESHOLD,
  ) -> RemovalResult:
    image = to_bgr(image)
    result = image.copy()
    detection = self.detect(image, force_size=force_size)

    if not detection.detected and not force:
      return RemovalResult(
        image=result,
        removed=False,
        provider=self.name,
        skipped_reason="no_watermark_detected",
        confidence=detection.confidence,
      )

    if detection.confidence < min_confidence and not force:
      return RemovalResult(
        image=result,
        removed=False,
        provider=self.name,
        confidence=detection.confidence,
        skipped_reason="confidence_below_threshold",
      )

    size = force_size or detection.size or watermark_size_for(image.shape[1], image.shape[0])
    region = detection.region if detection.region is not None else self._catalog_region(
      image.shape[1], image.shape[0], size
    )

    alpha = self._alpha_for(size)
    if alpha.shape[0] != region.width:
      alpha = cv2.resize(alpha, (region.width, region.height), interpolation=cv2.INTER_LINEAR)

    roi = result[region.y : region.y + region.height, region.x : region.x + region.width]
    cleaned = reverse_alpha_blend(roi, alpha, logo_value=self.logo_value)
    result[region.y : region.y + region.height, region.x : region.x + region.width] = cleaned

    return RemovalResult(
      image=result,
      removed=True,
      provider=self.name,
      region=region,
      confidence=detection.confidence,
    )

  def add_watermark(
    self,
    image: np.ndarray,
    *,
    force_size: WatermarkSize | None = None,
  ) -> np.ndarray:
    """Apply Gemini-style watermark (for tests and round-trip validation)."""
    image = to_bgr(image)
    result = image.copy()
    h, w = image.shape[:2]
    size = force_size or watermark_size_for(w, h)
    region = LAYOUTS[size].region(w, h)
    alpha = self._alpha_for(size)
    roi = result[region.y : region.y + region.height, region.x : region.x + region.width]
    cleaned = apply_watermark(roi, alpha, logo_value=self.logo_value)
    result[region.y : region.y + region.height, region.x : region.x + region.width] = cleaned
    return result
