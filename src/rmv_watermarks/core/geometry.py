from __future__ import annotations

from dataclasses import dataclass

from rmv_watermarks.types import WatermarkRegion, WatermarkSize


@dataclass
class SparkleLayout:
  logo_size: int
  margin: int

  def position(self, image_width: int, image_height: int) -> tuple[int, int]:
    x = image_width - self.logo_size - self.margin
    y = image_height - self.logo_size - self.margin
    return x, y

  def region(self, image_width: int, image_height: int) -> WatermarkRegion:
    x, y = self.position(image_width, image_height)
    return WatermarkRegion(x=x, y=y, width=self.logo_size, height=self.logo_size)


# Gemini/Nano Banana bottom-right sparkle layout.
LAYOUTS: dict[WatermarkSize, SparkleLayout] = {
  WatermarkSize.SMALL: SparkleLayout(logo_size=48, margin=32),
  WatermarkSize.LARGE: SparkleLayout(logo_size=96, margin=64),
}


def watermark_size_for(image_width: int, image_height: int) -> WatermarkSize:
  """Both dimensions must exceed 1024 for the large watermark."""
  if image_width > 1024 and image_height > 1024:
    return WatermarkSize.LARGE
  return WatermarkSize.SMALL


def layout_for(image_width: int, image_height: int) -> SparkleLayout:
  return LAYOUTS[watermark_size_for(image_width, image_height)]
