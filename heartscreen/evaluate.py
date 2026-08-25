"""Stratified k-fold cross-validation scored with the CinC 2017 challenge metric."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from heartscreen.data import LABELS, CachedDataset, build_cache
from heartscreen.train import Config, load_config, train_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def challenge_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    """Mean F1 over Normal, AF, and Other, the official challenge score."""
    per_class = {}
    for c, name in enumerate(LABELS):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        denom = int(np.sum(y_true == c) + np.sum(y_pred == c))
        per_class[name] = 2 * tp / denom if denom else 0.0
    return (per_class["N"] + per_class["A"] + per_class["O"]) / 3, per_class


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t, p] += 1
    return matrix


def plot_confusion(matrix: np.ndarray, path: str | Path) -> None:
    rows = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if rows[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{rows[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_cv(cfg: Config) -> dict:
    """Run stratified k-fold CV and write per-fold logs, params, and a summary."""
    cache = Path(cfg.cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        build_cache(cfg.data_dir, cache)
        print(f"built cache in {time.perf_counter() - start:.0f}s")
    dataset = CachedDataset(cache)

    indices = np.arange(len(dataset))
    labels = dataset.labels
    if cfg.limit_records is not None and cfg.limit_records < len(indices):
        indices, _ = train_test_split(
            indices, train_size=cfg.limit_records, stratify=labels, random_state=cfg.seed
        )
        labels = labels[indices]

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = lambda yt, yp: challenge_f1(yt, yp)[0]  # noqa: E731

    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    fold_scores, fold_per_class = [], []
    pooled_true, pooled_pred = [], []
    wall_start = time.perf_counter()
    for k, (tr, va) in enumerate(skf.split(indices, labels)):
        tr_idx, va_idx = indices[tr], indices[va]
        result = train_fold(
            cfg,
            [dataset.signal(i) for i in tr_idx],
            dataset.labels[tr_idx],
            [dataset.signal(i) for i in va_idx],
            dataset.labels[va_idx],
            out_dir / f"fold{k}",
            metric,
        )
        y_true = dataset.labels[va_idx]
        y_pred = result["val_logits"].argmax(1)
        score, per_class = challenge_f1(y_true, y_pred)
        fold_scores.append(score)
        fold_per_class.append(per_class)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        print(
            f"fold {k}: challenge F1 {score:.4f}  "
            + "  ".join(f"{n} {v:.3f}" for n, v in per_class.items())
        )

    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    pooled_score, pooled_per_class = challenge_f1(y_true, y_pred)
    summary = {
        "folds": cfg.folds,
        "records": int(len(indices)),
        "fold_scores": [round(s, 4) for s in fold_scores],
        "mean_f1": round(float(np.mean(fold_scores)), 4),
        "std_f1": round(float(np.std(fold_scores)), 4),
        "pooled_f1": round(pooled_score, 4),
        "pooled_per_class": {k: round(v, 4) for k, v in pooled_per_class.items()},
        "per_fold_per_class": [{k: round(v, 4) for k, v in d.items()} for d in fold_per_class],
        "wall_seconds": round(time.perf_counter() - wall_start, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_confusion(confusion(y_true, y_pred), out_dir / "confusion.png")
    print(
        f"mean challenge F1 {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f} "
        f"({summary['wall_seconds']:.0f}s)"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true", help="run the reduced smoke configuration")
    args = parser.parse_args()
    cfg = load_config("configs/smoke.yaml" if args.smoke else args.config)
    run_cv(cfg)


if __name__ == "__main__":
    main()
