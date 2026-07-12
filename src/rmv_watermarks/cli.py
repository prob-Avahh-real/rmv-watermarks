from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from rmv_watermarks import __version__
from rmv_watermarks.api import detect, list_providers, remove
from rmv_watermarks.types import WatermarkSize


def _parse_size(value: str | None) -> WatermarkSize | None:
  if value is None:
    return None
  mapping = {"small": WatermarkSize.SMALL, "large": WatermarkSize.LARGE}
  try:
    return mapping[value.lower()]
  except KeyError as exc:
    raise click.BadParameter("size must be 'small' or 'large'") from exc


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output image path")
@click.option(
  "-p",
  "--provider",
  default="auto",
  show_default=True,
  help="Watermark provider or 'auto'",
)
@click.option("--force", is_flag=True, help="Remove even when detection is uncertain")
@click.option("--detect-only", is_flag=True, help="Only run detection, do not modify")
@click.option("--size", type=click.Choice(["small", "large"]), help="Force watermark size")
@click.option("--min-confidence", type=float, default=0.35, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout")
@click.option("-q", "--quiet", is_flag=True)
@click.version_option(__version__, prog_name="rmv-watermarks")
def main(
  input: Path,
  output: Path | None,
  provider: str,
  force: bool,
  detect_only: bool,
  size: str | None,
  min_confidence: float,
  as_json: bool,
  quiet: bool,
) -> None:
  """Remove visible AI watermarks from an image."""
  force_size = _parse_size(size)

  if detect_only:
    detect_provider = "gemini" if provider == "auto" else provider
    result = detect(input, provider=detect_provider, force_size=force_size)
    payload = {
      "detected": result.detected,
      "confidence": result.confidence,
      "provider": result.provider,
      "region": result.region.as_tuple if result.region else None,
      "size": result.size.value if result.size else None,
      "details": result.details,
    }
    if as_json:
      click.echo(json.dumps(payload, indent=2))
    elif not quiet:
      if result.detected:
        click.echo(f"Detected ({result.provider}) confidence={result.confidence:.3f}")
      else:
        click.echo("No watermark detected")
    sys.exit(0 if result.detected else 1)

  out_path = output or input
  result = remove(
    input,
    out_path,
    provider=provider,
    force=force,
    force_size=force_size,
    min_confidence=min_confidence,
  )

  payload = {
    "removed": result.removed,
    "provider": result.provider,
    "confidence": result.confidence,
    "region": result.region.as_tuple if result.region else None,
    "output": str(out_path),
    "skipped_reason": result.skipped_reason,
  }

  if as_json:
    click.echo(json.dumps(payload, indent=2))
  elif not quiet:
    if result.removed:
      click.echo(f"Removed watermark via {result.provider} -> {out_path}")
    else:
      reason = result.skipped_reason or "unknown"
      click.echo(f"No changes ({reason})")

  if provider == "auto" and not quiet and not as_json:
    click.echo(f"Providers: {', '.join(list_providers())}")

  sys.exit(0 if result.removed or out_path.exists() else 1)


if __name__ == "__main__":
  main()
