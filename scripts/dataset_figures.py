"""Figures for the CinC 2017 training set and for the long-recording screening funnel.

Run all of them with `uv run python scripts/dataset_figures.py`, or a single
group with `--figures dataset` / `--figures funnel`. The dataset figures read
the downloaded CinC 2017 records; the funnel reads the screening run outputs in
results/screening/, so every plotted number comes from a file on disk.
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import wfdb

from heartscreen.data import LABEL_TO_INDEX, LABELS, load_reference
from heartscreen.preprocessing import FS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cinc2017"
SCREENING_DIR = ROOT / "results" / "screening"
FIG_DIR = ROOT / "docs" / "figures"

AF_CLASS = LABEL_TO_INDEX["A"]
WINDOW_S = 30.0
STRIDE_S = 15.0
THRESHOLD = 0.5

BAND_COLOR = "#dde4ec"
BAND_TEXT = "#20415e"
STAGE_COLOR = "#3b6ea5"
MERGE_COLOR = "#20415e"
OUTPUT_COLOR = "#d95f02"


def dataset_figures() -> dict:
    """Class balance and record length of the CinC 2017 public training set."""
    records, labels = load_reference(DATA_DIR)
    lengths = np.array([wfdb.rdheader(str(DATA_DIR / "training2017" / r)).sig_len for r in records])
    seconds = lengths / FS
    counts = np.bincount(labels, minlength=4)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    bars = ax.bar(LABELS, counts, color=STAGE_COLOR)
    ax.bar_label(bars, [f"{c}\n{c / len(labels):.1%}" for c in counts], fontsize=8)
    ax.set_ylabel("records")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(seconds, bins=52, color=STAGE_COLOR)
    ax.set_xlabel("record length (s)")
    ax.set_ylabel("records")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_histogram.png", dpi=150)
    plt.close(fig)

    return {
        "records": len(records),
        "class_counts": {name: int(c) for name, c in zip(LABELS, counts, strict=True)},
        "length_seconds": {
            "min": round(float(seconds.min()), 1),
            "median": round(float(np.median(seconds)), 1),
            "mean": round(float(seconds.mean()), 1),
            "max": round(float(seconds.max()), 1),
        },
    }


def funnel_stats(screening_dir: Path) -> dict:
    """Every stage count of the screening run, read from its own output files."""
    windows = pd.read_csv(screening_dir / "windows.csv")
    episodes = pd.read_csv(screening_dir / "candidates.csv")

    per_record_seconds = windows.groupby("record")["start_s"].max() + WINDOW_S
    hours = float(per_record_seconds.sum() / 3600)
    n_windows = int(len(windows))
    n_flagged = int((windows["p_af"] >= THRESHOLD).sum())
    n_episodes = int(len(episodes))
    n_vetted = int(episodes["vet_pass"].sum())

    scored = windows.dropna(subset=["af_frac"])
    truth = scored["af_frac"] >= 0.5
    pred = scored["pred"] == AF_CLASS
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())

    return {
        "records": int(per_record_seconds.size),
        "hours": hours,
        "shortest_record_h": float(per_record_seconds.min() / 3600),
        "longest_record_h": float(per_record_seconds.max() / 3600),
        "windows": n_windows,
        "window_coverage": n_windows * STRIDE_S / 3600 / hours,
        "flagged_windows": n_flagged,
        "episodes": n_episodes,
        "vetted": n_vetted,
        "scored_against_annotations": int(len(scored)),
        "af_prevalence": float(truth.mean()),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
    }


def screening_funnel(screening_dir: Path = SCREENING_DIR) -> dict:
    """Stage-by-stage funnel from recording-hours to vetted, ranked AF candidates."""
    s = funnel_stats(screening_dir)

    stages = [
        (
            "recordings\nscreened",
            s["windows"],
            f"{s['hours']:,.0f} recording-hours",
            f"{s['records']} Holter recordings, {s['shortest_record_h']:.1f} to "
            f"{s['longest_record_h']:.1f} h each",
            BAND_COLOR,
        ),
        (
            "windows\nscored",
            s["windows"],
            f"{s['windows']:,} windows",
            f"{s['window_coverage']:.0%} of recorded time, 30 s window at 15 s stride",
            STAGE_COLOR,
        ),
        (
            f"windows above\np(AF) {THRESHOLD:.1f}",
            s["flagged_windows"],
            f"{s['flagged_windows']:,} windows",
            f"{s['flagged_windows'] / s['windows']:.1%} of the windows scored",
            STAGE_COLOR,
        ),
        (
            "candidate\nepisodes",
            s["episodes"],
            f"{s['episodes']:,} episodes",
            f"{s['episodes'] / s['flagged_windows']:.1%} as many items after merging runs",
            MERGE_COLOR,
        ),
        (
            "vetted\ncandidates",
            s["vetted"],
            f"{s['vetted']:,} candidates",
            f"{s['vetted'] / s['episodes']:.1%} of episodes pass the quality and RR gates",
            OUTPUT_COLOR,
        ),
    ]

    xmax = 10 ** np.ceil(np.log10(max(v for _, v, _, _, _ in stages)))
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.855, bottom=0.135)
    ax.set_xscale("log")
    ax.set_xlim(1, xmax)

    def stage_text(y: float, value: float, headline: str, retained: str, color: str) -> None:
        ink = BAND_TEXT if color == BAND_COLOR else "white"
        ax.barh(y, value - 1, left=1, height=0.72, color=color)
        ax.text(
            1.6,
            y - 0.14,
            headline,
            va="center",
            ha="left",
            color=ink,
            fontsize=13.5,
            weight="bold",
        )
        ax.text(1.6, y + 0.19, retained, va="center", ha="left", color=ink, fontsize=10)

    for y, (_, value, headline, retained, color) in enumerate(stages):
        stage_text(y, value, headline, retained, color)

    outcome_y = len(stages) + 0.35
    stage_text(
        outcome_y,
        xmax,
        f"sensitivity {s['sensitivity']:.3f}    specificity {s['specificity']:.3f}",
        f"PPV {s['ppv']:.3f} on {s['scored_against_annotations']:,} windows scored against "
        "the rhythm annotations",
        BAND_COLOR,
    )

    ax.set_yticks([*range(len(stages)), outcome_y])
    ax.set_yticklabels(
        [name for name, _, _, _, _ in stages] + ["window-level\nagreement"], fontsize=11
    )
    ax.set_ylim(outcome_y + 0.55, -0.6)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.tick_params(axis="both", which="both", length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    fig.text(
        0.012,
        0.968,
        f"{s['hours']:,.0f} recording-hours reduce to {s['vetted']:,} vetted AF candidates "
        f"at {s['sensitivity']:.3f} window sensitivity",
        fontsize=13.5,
        weight="bold",
        va="top",
    )
    fig.text(
        0.012,
        0.915,
        "MIT-BIH Long-Term AF Database. Solid bars are item counts on a log scale; the shaded "
        "bands are the input and the measured outcome.",
        fontsize=9.5,
        color="#444444",
        va="top",
    )
    fig.text(
        0.012,
        0.085,
        textwrap.fill(
            f"LTAF is an AF-enriched cohort: {s['af_prevalence']:.1%} of scored windows are "
            "annotated AF, so this sensitivity and specificity do not transfer to a "
            "low-prevalence screening population. Adjacent windows overlap by 15 s, which "
            f"correlates the {s['scored_against_annotations']:,} window scores and shrinks the "
            "effective sample size.",
            width=118,
        ),
        fontsize=9,
        color="#444444",
        va="top",
    )

    fig.savefig(FIG_DIR / "screening_funnel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", choices=["all", "dataset", "funnel"], default="all")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    wanted_dataset = args.figures in ("all", "dataset")
    wanted_funnel = args.figures in ("all", "funnel")

    if wanted_dataset:
        if (DATA_DIR / "training2017").exists():
            stats["dataset"] = dataset_figures()
        elif args.figures == "dataset":
            raise SystemExit(
                f"missing {DATA_DIR / 'training2017'}; run ./scripts/download_cinc2017.sh"
            )
        else:
            print(f"skipping dataset figures: {DATA_DIR / 'training2017'} not found")

    if wanted_funnel:
        if (SCREENING_DIR / "windows.csv").exists():
            stats["screening_funnel"] = screening_funnel()
        elif args.figures == "funnel":
            raise SystemExit(
                f"missing {SCREENING_DIR / 'windows.csv'}; run python -m heartscreen.screening"
            )
        else:
            print(f"skipping screening funnel: {SCREENING_DIR / 'windows.csv'} not found")

    print(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
