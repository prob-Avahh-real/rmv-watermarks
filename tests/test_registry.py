from __future__ import annotations

import pytest

from rmv_watermarks import registry
from rmv_watermarks.protocol import WatermarkProvider
from rmv_watermarks.providers.gemini import GeminiProvider


def test_default_provider_registered() -> None:
  names = registry.list_names()
  assert "gemini" in names


def test_get_unknown_provider_raises() -> None:
  with pytest.raises(KeyError, match="Unknown provider"):
    registry.get("nonexistent")


def test_custom_registration() -> None:
  class DummyProvider(WatermarkProvider):
    name = "dummy"

    def detect(self, image, *, force_size=None):
      from rmv_watermarks.types import DetectionResult

      return DetectionResult(detected=False, confidence=0.0, provider=self.name)

    def remove(self, image, *, force=False, force_size=None, min_confidence=0.35):
      from rmv_watermarks.types import RemovalResult

      return RemovalResult(image=image, removed=False, provider=self.name)

  registry.register(DummyProvider())
  assert "dummy" in registry.list_names()
  assert registry.get("dummy").name == "dummy"

  # Re-register gemini to avoid polluting other tests
  registry.register(GeminiProvider())
