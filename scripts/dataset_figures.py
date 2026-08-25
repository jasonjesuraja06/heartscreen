"""Class distribution and record length figures for the CinC 2017 training set."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import wfdb

from heartscreen.data import LABELS, load_reference
from heartscreen.preprocessing import FS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cinc2017"
FIG_DIR = ROOT / "docs" / "figures"


def main() -> None:
    records, labels = load_reference(DATA_DIR)
    lengths = np.array(
        [wfdb.rdheader(str(DATA_DIR / "training2017" / r)).sig_len for r in records]
    )
    seconds = lengths / FS
    counts = np.bincount(labels, minlength=4)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    bars = ax.bar(LABELS, counts, color="#3b6ea5")
    ax.bar_label(bars, [f"{c}\n{c / len(labels):.1%}" for c in counts], fontsize=8)
    ax.set_ylabel("records")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(seconds, bins=52, color="#3b6ea5")
    ax.set_xlabel("record length (s)")
    ax.set_ylabel("records")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_histogram.png", dpi=150)

    stats = {
        "records": len(records),
        "class_counts": {name: int(c) for name, c in zip(LABELS, counts, strict=True)},
        "length_seconds": {
            "min": round(float(seconds.min()), 1),
            "median": round(float(np.median(seconds)), 1),
            "mean": round(float(seconds.mean()), 1),
            "max": round(float(seconds.max()), 1),
        },
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
