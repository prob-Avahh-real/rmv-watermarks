from __future__ import annotations

from pathlib import Path

import numpy as np

from rmv_watermarks import registry
from rmv_watermarks.core.io import decode_bytes, encode_bytes, imread, imwrite, to_bgr
from rmv_watermarks.types import DetectionResult, RemovalResult, WatermarkSize

ImageInput = str | Path | bytes | np.ndarray


def _load_image(source: ImageInput) -> np.ndarray:
  if isinstance(source, np.ndarray):
    return to_bgr(source)
  if isinstance(source, bytes):
    decoded = decode_bytes(source)
    if decoded is None:
      raise ValueError("Failed to decode image bytes")
    return to_bgr(decoded)
  path = Path(source)
  loaded = imread(path)
  if loaded is None:
    raise FileNotFoundError(f"Cannot read image: {path}")
  return to_bgr(loaded)


def _save_image(image: np.ndarray, dest: str | Path) -> None:
  if not imwrite(dest, image):
    raise OSError(f"Failed to write image: {dest}")


def list_providers() -> list[str]:
  return registry.list_names()


def detect(
  source: ImageInput,
  *,
  provider: str = "gemini",
  force_size: WatermarkSize | None = None,
) -> DetectionResult:
  image = _load_image(source)
  return registry.get(provider).detect(image, force_size=force_size)


def remove(
  source: ImageInput,
  dest: str | Path | None = None,
  *,
  provider: str = "auto",
  force: bool = False,
  force_size: WatermarkSize | None = None,
  min_confidence: float = 0.35,
  return_bytes: bool = False,
  output_ext: str = ".png",
) -> RemovalResult | tuple[RemovalResult, bytes]:
  image = _load_image(source)
  providers = registry.list_names() if provider == "auto" else [provider]

  last: RemovalResult | None = None
  for name in providers:
    result = registry.get(name).remove(
      image,
      force=force,
      force_size=force_size,
      min_confidence=min_confidence,
    )
    last = result
    if result.removed:
      if dest is not None:
        _save_image(result.image, dest)
      if return_bytes:
        return result, encode_bytes(result.image, output_ext)
      return result

  assert last is not None
  if dest is not None:
    _save_image(last.image, dest)
  if return_bytes:
    return last, encode_bytes(last.image, output_ext)
  return last
