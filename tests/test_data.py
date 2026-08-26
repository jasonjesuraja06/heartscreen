import numpy as np
import pytest

from heartscreen.data import LABEL_TO_INDEX, load_reference, load_signal
from tests.conftest import DATA_DIR, requires_data


def test_label_parsing(tmp_path):
    (tmp_path / "training2017").mkdir()
    (tmp_path / "training2017" / "REFERENCE.csv").write_text(
        "A00001,N\nA00002,A\nA00003,O\nA00004,~\n"
    )
    with pytest.warns(UserWarning, match="REFERENCE-v3"):
        records, labels = load_reference(tmp_path)
    assert records == ["A00001", "A00002", "A00003", "A00004"]
    assert labels.tolist() == [0, 1, 2, 3]
    assert labels.dtype == np.int32


def test_v3_reference_preferred(tmp_path):
    (tmp_path / "training2017").mkdir()
    (tmp_path / "training2017" / "REFERENCE.csv").write_text("A00001,N\n")
    (tmp_path / "REFERENCE-v3.csv").write_text("A00001,A\n")
    _, labels = load_reference(tmp_path)
    assert labels.tolist() == [LABEL_TO_INDEX["A"]]


@requires_data
def test_record_count():
    records, labels = load_reference(DATA_DIR)
    assert len(records) == 8528
    assert len(labels) == 8528
    assert set(np.unique(labels)) <= {0, 1, 2, 3}


@requires_data
def test_waveform_shape():
    signal = load_signal(DATA_DIR, "A00001")
    assert signal.ndim == 1
    assert signal.dtype == np.float32
    # Record lengths span 9 to 61 seconds at 300 Hz.
    assert 2700 <= len(signal) <= 18400
