from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from rmv_watermarks.types import DetectionResult, RemovalResult, WatermarkSize


class WatermarkProvider(ABC):
  """Plugin interface for a specific visible watermark family."""

  name: ClassVar[str]

  @abstractmethod
  def detect(
    self,
    image: np.ndarray,
    *,
    force_size: WatermarkSize | None = None,
  ) -> DetectionResult:
    raise NotImplementedError

  @abstractmethod
  def remove(
    self,
    image: np.ndarray,
    *,
    force: bool = False,
    force_size: WatermarkSize | None = None,
    min_confidence: float = 0.35,
  ) -> RemovalResult:
    raise NotImplementedError
