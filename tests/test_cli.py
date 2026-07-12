from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from rmv_watermarks.cli import main


def test_cli_remove(tmp_path: Path, small_watermarked) -> None:
  import cv2

  src = tmp_path / "in.png"
  dst = tmp_path / "out.png"
  cv2.imwrite(str(src), small_watermarked)

  runner = CliRunner()
  result = runner.invoke(main, [str(src), "-o", str(dst), "--force", "-q"])
  assert result.exit_code == 0
  assert dst.exists()


def test_cli_detect_only_json(tmp_path: Path, small_watermarked) -> None:
  import cv2

  src = tmp_path / "in.png"
  cv2.imwrite(str(src), small_watermarked)

  runner = CliRunner()
  result = runner.invoke(main, [str(src), "--detect-only", "--json"])
  assert result.exit_code == 0
  payload = json.loads(result.output)
  assert payload["detected"] is True
  assert payload["confidence"] > 0


def test_cli_detect_only_clean_exits_nonzero(tmp_path: Path, gradient_image) -> None:
  import cv2

  src = tmp_path / "clean.png"
  cv2.imwrite(str(src), gradient_image)

  runner = CliRunner()
  result = runner.invoke(main, [str(src), "--detect-only", "-q"])
  assert result.exit_code == 1


def test_cli_help() -> None:
  runner = CliRunner()
  result = runner.invoke(main, ["--help"])
  assert result.exit_code == 0
  assert "Remove visible AI watermarks" in result.output
