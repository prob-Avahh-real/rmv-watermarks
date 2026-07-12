from __future__ import annotations

from rmv_watermarks.protocol import WatermarkProvider

_PROVIDERS: dict[str, WatermarkProvider] = {}


def _ensure_defaults() -> None:
  if "gemini" not in _PROVIDERS:
    from rmv_watermarks.providers.gemini import GeminiProvider

    register(GeminiProvider())


def register(provider: WatermarkProvider) -> None:
  _PROVIDERS[provider.name] = provider


def get(name: str) -> WatermarkProvider:
  _ensure_defaults()
  try:
    return _PROVIDERS[name]
  except KeyError as exc:
    available = ", ".join(sorted(_PROVIDERS)) or "(none)"
    raise KeyError(f"Unknown provider '{name}'. Available: {available}") from exc


def list_names() -> list[str]:
  _ensure_defaults()
  return sorted(_PROVIDERS)
