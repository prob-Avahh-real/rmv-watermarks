from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from rmv_watermarks.cli import main


def test_cli_remove_verbose(tmp_path: Path, small_watermarked) -> None:
  import cv2

  src = tmp_path / "in.png"
  dst = tmp_path / "out.png"
  cv2.imwrite(str(src), small_watermarked)

  runner = CliRunner()
  result = runner.invoke(main, [str(src), "-o", str(dst), "--force"])
  assert result.exit_code == 0
  assert "Removed watermark" in result.output


def test_cli_no_changes_message(tmp_path: Path, gradient_image) -> None:
  import cv2

  src = tmp_path / "clean.png"
  cv2.imwrite(str(src), gradient_image)

  runner = CliRunner()
  result = runner.invoke(main, [str(src), "-o", str(tmp_path / "out.png")])
  assert result.exit_code == 0
  assert "No changes" in result.output


def test_cli_detect_only_text(tmp_path: Path, small_watermarked) -> None:
  import cv2

  src = tmp_path / "in.png"
  cv2.imwrite(str(src), small_watermarked)

  runner = CliRunner()
  result = runner.invoke(main, [str(src), "--detect-only"])
  assert result.exit_code == 0
  assert "Detected" in result.output


def test_cli_invalid_size(tmp_path: Path, small_watermarked) -> None:
  import cv2

  src = tmp_path / "in.png"
  cv2.imwrite(str(src), small_watermarked)

  runner = CliRunner()
  result = runner.invoke(main, [str(src), "--size", "invalid"])
  assert result.exit_code != 0
