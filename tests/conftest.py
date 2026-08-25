from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cinc2017"

requires_data = pytest.mark.skipif(
    not (DATA_DIR / "training2017").exists(),
    reason="CinC 2017 data not downloaded",
)
