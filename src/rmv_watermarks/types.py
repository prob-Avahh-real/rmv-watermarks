from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class WatermarkSize(str, Enum):
  SMALL = "small"
  LARGE = "large"


@dataclass(frozen=True, slots=True)
class WatermarkRegion:
  x: int
  y: int
  width: int
  height: int

  @property
  def as_tuple(self) -> tuple[int, int, int, int]:
    return self.x, self.y, self.width, self.height


@dataclass(frozen=True, slots=True)
class DetectionResult:
  detected: bool
  confidence: float
  provider: str
  region: WatermarkRegion | None = None
  size: WatermarkSize | None = None
  details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RemovalResult:
  image: Any
  removed: bool
  provider: str | None = None
  region: WatermarkRegion | None = None
  confidence: float = 0.0
  skipped_reason: str | None = None
