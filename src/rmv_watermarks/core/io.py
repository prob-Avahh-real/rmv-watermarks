from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


def to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
  """Normalize an image array to 3-channel BGR uint8."""
  if image.ndim == 2:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  if image.shape[2] == 4:
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
  if image.shape[2] == 3:
    return image
  raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def imread(path: str | Path) -> NDArray[np.uint8] | None:
  """Unicode-safe cv2.imread."""
  try:
    data = np.fromfile(str(path), dtype=np.uint8)
  except OSError:
    return None
  if data.size == 0:
    return None
  return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite(path: str | Path, image: NDArray[np.uint8]) -> bool:
  """Unicode-safe cv2.imwrite."""
  suffix = Path(path).suffix.lower()
  ext = suffix if suffix else ".png"
  ok, encoded = cv2.imencode(ext, image)
  if not ok:
    return False
  encoded.tofile(str(path))
  return True


def decode_bytes(data: bytes) -> NDArray[np.uint8] | None:
  arr = np.frombuffer(data, dtype=np.uint8)
  if arr.size == 0:
    return None
  return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def encode_bytes(image: NDArray[np.uint8], ext: str = ".png") -> bytes:
  ok, encoded = cv2.imencode(ext, image)
  if not ok:
    raise ValueError(f"Failed to encode image as {ext}")
  return encoded.tobytes()
