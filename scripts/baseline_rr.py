"""RR-interval and signal-quality baseline: logistic regression under the CV protocol.

Anchors the CNN score by measuring how far the hand-built screening features
(beat rate, RR irregularity, spectral quality) get on the same task, same
folds, same metric.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from heartscreen.data import CachedDataset
from heartscreen.evaluate import challenge_f1
from heartscreen.preprocessing import FS
from heartscreen.screening import rr_metrics, window_sqi

CACHE = Path("results/baseline/features.csv")


def extract_features(dataset: CachedDataset) -> pd.DataFrame:
    rows = []
    for i in range(len(dataset)):
        signal = dataset.signal(i)
        rr = rr_metrics(signal)
        sqi = window_sqi(signal)
        rows.append(
            {
                "beats_per_s": rr["beats"] / (len(signal) / FS),
                "cv_rr": rr["cv_rr"],
                "rmssd_ms": rr["rmssd_ms"],
                "flatline_frac": sqi["flatline_frac"],
                "qrs_band_ratio": sqi["qrs_band_ratio"],
                "label": int(dataset.labels[i]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    dataset = CachedDataset("data/cache/cinc2017.npz")
    if CACHE.exists():
        features = pd.read_csv(CACHE)
    else:
        start = time.perf_counter()
        features = extract_features(dataset)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(CACHE, index=False)
        print(f"features extracted in {time.perf_counter() - start:.0f}s")

    x = features.drop(columns="label").fillna(0.0).to_numpy()
    y = features["label"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train, val in skf.split(x, y):
        model.fit(x[train], y[train])
        score, per_class = challenge_f1(y[val], model.predict(x[val]))
        scores.append(score)
        print(f"fold: {score:.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in per_class.items()))
    print(f"baseline mean challenge F1 {np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
