"""CinC 2017 record loading, labels, and a preprocessed signal cache."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from heartscreen.preprocessing import FS, condition

LABELS = ("N", "A", "O", "~")
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def load_reference(data_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Return record names and integer labels, preferring the revised v3 reference."""
    data_dir = Path(data_dir)
    v3 = data_dir / "REFERENCE-v3.csv"
    if not v3.exists():
        warnings.warn("REFERENCE-v3.csv not found, using original challenge labels", stacklevel=2)
    path = v3 if v3.exists() else data_dir / "training2017" / "REFERENCE.csv"
    ref = pd.read_csv(path, header=None, names=["record", "label"])
    labels = ref["label"].map(LABEL_TO_INDEX).to_numpy(np.int32)
    return ref["record"].tolist(), labels


def load_signal(data_dir: str | Path, record: str) -> np.ndarray:
    """Read one record's waveform in mV as float32, verifying the 300 Hz rate."""
    signal, fields = wfdb.rdsamp(str(Path(data_dir) / "training2017" / record))
    if fields["fs"] != FS:
        raise ValueError(f"{record}: expected {FS} Hz, got {fields['fs']}")
    return signal[:, 0].astype(np.float32)


def build_cache(data_dir: str | Path, cache_path: str | Path) -> None:
    """Filter and normalize every record once, storing ragged signals in one npz."""
    records, labels = load_reference(data_dir)
    signals = [condition(load_signal(data_dir, r)) for r in records]
    lengths = np.array([len(s) for s in signals], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez(
        cache_path,
        flat=np.concatenate(signals),
        offsets=offsets,
        labels=labels,
        records=np.array(records),
    )


class CachedDataset:
    """Ragged store of conditioned signals with O(1) per-record access."""

    def __init__(self, cache_path: str | Path):
        data = np.load(cache_path, allow_pickle=False)
        self.flat = data["flat"]
        self.offsets = data["offsets"]
        self.labels = data["labels"]
        self.records = [str(r) for r in data["records"]]

    def __len__(self) -> int:
        return len(self.labels)

    def signal(self, i: int) -> np.ndarray:
        return self.flat[self.offsets[i] : self.offsets[i + 1]]
