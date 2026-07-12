from __future__ import annotations

from rmv_watermarks.core.geometry import LAYOUTS, layout_for, watermark_size_for
from rmv_watermarks.types import WatermarkSize


def test_small_size_at_or_below_1024() -> None:
  assert watermark_size_for(1024, 1024) == WatermarkSize.SMALL
  assert watermark_size_for(800, 600) == WatermarkSize.SMALL


def test_large_size_requires_both_dimensions() -> None:
  assert watermark_size_for(1025, 800) == WatermarkSize.SMALL
  assert watermark_size_for(1200, 1100) == WatermarkSize.LARGE


def test_layout_position_small() -> None:
  layout = LAYOUTS[WatermarkSize.SMALL]
  x, y = layout.position(800, 600)
  assert (x, y) == (800 - 48 - 32, 600 - 48 - 32)


def test_layout_position_large() -> None:
  layout = LAYOUTS[WatermarkSize.LARGE]
  x, y = layout.position(1600, 1200)
  assert (x, y) == (1600 - 96 - 64, 1200 - 96 - 64)


def test_layout_for_matches_size() -> None:
  assert layout_for(500, 500).logo_size == 48
  assert layout_for(2000, 2000).logo_size == 96
