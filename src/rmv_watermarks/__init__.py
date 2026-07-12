"""rmv-watermarks: lightweight, pluggable visible watermark removal."""

from rmv_watermarks.api import detect, list_providers, remove
from rmv_watermarks.types import RemovalResult, WatermarkRegion

__all__ = ["RemovalResult", "WatermarkRegion", "detect", "list_providers", "remove"]
__version__ = "0.1.0"
