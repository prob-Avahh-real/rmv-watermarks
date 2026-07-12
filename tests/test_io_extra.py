from __future__ import annotations

import numpy as np
import pytest

from rmv_watermarks.core.io import to_bgr


def test_to_bgr_unsupported_channels() -> None:
  image = np.zeros((4, 4, 2), dtype=np.uint8)
  with pytest.raises(ValueError, match="Unsupported channel"):
    to_bgr(image)
