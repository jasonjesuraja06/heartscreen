"""Long-recording AF screening: sliding-window inference, vetting, ranked candidates."""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import sosfiltfilt
from wfdb import processing

from heartscreen.preprocessing import FS, bandpass_sos, resample_to_fs
from heartscreen.train import Config, build_model, load_config, load_params, make_steps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AF_CLASS = 1
NOISE_CLASS = 3
AF_RHYTHMS = {"(AFIB", "(AFL"}


@dataclasses.dataclass
class Episode:
    record: str
    start_s: float
    end_s: float
    mean_p_af: float
    max_p_af: float
    n_windows: int


def rhythm_intervals(annotation, sig_len: int) -> list[tuple[int, int, str]]:
    """Convert rhythm change markers into (start, end, rhythm) sample intervals."""
    changes = [
        (s, aux.rstrip("\x00"))
        for s, aux in zip(annotation.sample, annotation.aux_note, strict=True)
        if aux.startswith("(")
    ]
    intervals = []
    for i, (start, rhythm) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else sig_len
        if end > start:
            intervals.append((int(start), int(end), rhythm))
    return intervals


def af_fraction(intervals: list[tuple[int, int, str]], start: int, end: int) -> float:
    overlap = sum(max(0, min(end, e) - max(start, s)) for s, e, r in intervals if r in AF_RHYTHMS)
    return overlap / (end - start)


def window_sqi(window: np.ndarray) -> dict[str, float]:
    """Cheap per-window quality indices computed on the conditioned signal."""
    diffs = np.abs(np.diff(window))
    flatline = float(np.mean(diffs < 1e-4))
    spectrum = np.abs(np.fft.rfft(window)) ** 2
    freqs = np.fft.rfftfreq(len(window), 1 / FS)
    total = spectrum[freqs >= 0.5].sum()
    qrs_band = spectrum[(freqs >= 5) & (freqs <= 25)].sum()
    return {"flatline_frac": flatline, "qrs_band_ratio": float(qrs_band / max(total, 1e-12))}


def rr_metrics(window: np.ndarray) -> dict[str, float]:
    """R-peak based irregularity evidence from the XQRS detector."""
    peaks = processing.xqrs_detect(window.astype(np.float64), fs=FS, verbose=False)
    if len(peaks) < 3:
        return {"beats": float(len(peaks)), "cv_rr": np.nan, "rmssd_ms": np.nan}
    rr = np.diff(peaks) / FS
    return {
        "beats": float(len(peaks)),
        "cv_rr": float(rr.std() / rr.mean()),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000),
    }


def vet(sqi: dict[str, float], rr: dict[str, float], window_s: float) -> bool:
    """Heuristic plausibility check: usable signal and AF-like RR irregularity."""
    plausible_beats = 0.5 * window_s <= rr["beats"] <= 3.5 * window_s
    return bool(
        sqi["flatline_frac"] < 0.2
        and sqi["qrs_band_ratio"] > 0.3
        and plausible_beats
        and np.isfinite(rr["cv_rr"])
        and rr["cv_rr"] >= 0.10
    )


def sliding_probs(eval_step, params, signal: np.ndarray, cfg: Config, stride: int):
    """Softmax probabilities for every stride-spaced full window of a conditioned signal.

    The tail shorter than one window is not scored; recordings shorter than one
    window yield no windows at all.
    """
    length = cfg.window
    if len(signal) < length:
        return np.zeros(0, np.int64), np.zeros((0, 4), np.float32)
    starts = np.arange(0, len(signal) - length + 1, stride)
    probs = np.zeros((len(starts), 4), np.float32)
    bs = cfg.batch_size
    mask = np.ones((bs, length), np.float32)
    for lo in range(0, len(starts), bs):
        chunk = starts[lo : lo + bs]
        xb = np.stack([signal[s : s + length] for s in chunk])
        std = xb.std(axis=1, keepdims=True)
        xb = (xb - xb.mean(axis=1, keepdims=True)) / np.maximum(std, 1e-8)
        n_real = len(chunk)
        if n_real < bs:
            xb = np.concatenate([xb, np.zeros((bs - n_real, length), np.float32)])
        logits = np.asarray(eval_step(params, xb[..., None], mask))[:n_real]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs[lo : lo + n_real] = exp / exp.sum(axis=1, keepdims=True)
    return starts, probs


def find_episodes(record: str, starts, probs, window: int, threshold: float) -> list[Episode]:
    """Merge consecutive above-threshold AF windows into candidate episodes."""
    hot = probs[:, AF_CLASS] >= threshold
    episodes = []
    i = 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hot) and hot[j + 1]:
            j += 1
        p = probs[i : j + 1, AF_CLASS]
        episodes.append(
            Episode(
                record=record,
                start_s=starts[i] / FS,
                end_s=(starts[j] + window) / FS,
                mean_p_af=float(p.mean()),
                max_p_af=float(p.max()),
                n_windows=j - i + 1,
            )
        )
        i = j + 1
    return episodes


def screen_record(
    data_dir: Path, name: str, eval_step, params, cfg: Config, stride_s: int, threshold: float
):
    """Screen one recording; returns (windows df, episodes df, conditioned signal)."""
    rec = wfdb.rdrecord(str(data_dir / name), channels=[0])
    fs_in = rec.fs
    raw = np.nan_to_num(rec.p_signal[:, 0], nan=0.0)
    signal = resample_to_fs(raw, fs_in)
    signal = sosfiltfilt(bandpass_sos(), signal.astype(np.float64)).astype(np.float32)

    try:
        ann = wfdb.rdann(str(data_dir / name), "atr")
        intervals = rhythm_intervals(ann, rec.sig_len)
        scale = FS / fs_in
        intervals = [(int(s * scale), int(e * scale), r) for s, e, r in intervals]
    except FileNotFoundError:
        intervals = []

    stride = stride_s * FS
    starts, probs = sliding_probs(eval_step, params, signal, cfg, stride)
    windows = pd.DataFrame(
        {
            "record": name,
            "start_s": starts / FS,
            "p_af": probs[:, AF_CLASS],
            "p_noise": probs[:, NOISE_CLASS],
            "pred": probs.argmax(1),
        }
    )
    if intervals:
        windows["af_frac"] = [af_fraction(intervals, s, s + cfg.window) for s in starts]

    episodes = find_episodes(name, starts, probs, cfg.window, threshold)
    rows = []
    for ep in episodes:
        mid = int((ep.start_s + ep.end_s) / 2 * FS)
        lo = max(0, min(mid - cfg.window // 2, len(signal) - cfg.window))
        window = signal[lo : lo + cfg.window]
        sqi = window_sqi(window)
        rr = rr_metrics(window)
        rows.append(
            dataclasses.asdict(ep)
            | sqi
            | rr
            | {"vet_pass": vet(sqi, rr, cfg.window_seconds)}
            | (
                {"af_frac": af_fraction(intervals, int(ep.start_s * FS), int(ep.end_s * FS))}
                if intervals
                else {}
            )
        )
    return windows, pd.DataFrame(rows), signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--params", default="results/full/params.msgpack")
    parser.add_argument("--data-dir", default="data/ltafdb")
    parser.add_argument("--out-dir", default="results/screening")
    parser.add_argument("--stride-s", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="screen only the first N records")
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = load_params(cfg, args.params)
    _, eval_step = make_steps(build_model(cfg).apply, np.ones(4, np.float32))

    data_dir = Path(args.data_dir)
    names = [n for n in (data_dir / "RECORDS").read_text().split() if n]
    if args.limit:
        names = names[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_windows, all_episodes = [], []
    hours = 0.0
    start_time = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        windows, episodes, signal = screen_record(
            data_dir, name, eval_step, params, cfg, args.stride_s, args.threshold
        )
        hours += len(signal) / FS / 3600
        all_windows.append(windows)
        all_episodes.append(episodes)
        print(
            f"{name}: {len(signal) / FS / 3600:.1f} h, {len(episodes)} episodes "
            f"({time.perf_counter() - t0:.0f}s)"
        )
    minutes = (time.perf_counter() - start_time) / 60

    windows = pd.concat(all_windows, ignore_index=True)
    nonempty = [e for e in all_episodes if len(e)]
    episodes = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    if len(episodes):
        episodes = episodes.sort_values(
            ["vet_pass", "mean_p_af"], ascending=False, ignore_index=True
        )
    windows.to_csv(out_dir / "windows.csv", index=False)
    episodes.to_csv(out_dir / "candidates.csv", index=False)
    if len(episodes):
        plot_top_candidates(data_dir, episodes, out_dir / "top_candidates.png")

    print(
        f"\n{hours:.0f} recording-hours in {minutes:.1f} min "
        f"({hours / minutes:.1f} recording-hours/min)"
    )
    vetted = int(episodes["vet_pass"].sum()) if len(episodes) else 0
    print(f"{len(episodes)} candidate episodes, {vetted} pass vetting")

    if "af_frac" in windows:
        annotated = windows.dropna(subset=["af_frac"])
        truth = annotated["af_frac"] >= 0.5
        pred = annotated["pred"] == AF_CLASS
        tp = int((truth & pred).sum())
        tn = int((~truth & ~pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        print(
            f"window-level agreement vs rhythm annotations "
            f"(n={len(annotated)}): sensitivity {tp / max(tp + fn, 1):.3f}, "
            f"specificity {tn / max(tn + fp, 1):.3f}, ppv {tp / max(tp + fp, 1):.3f}"
        )


def plot_top_candidates(data_dir: Path, episodes: pd.DataFrame, path: Path, k: int = 10) -> None:
    """Plot a 10 s excerpt of each top-ranked episode with detected R peaks."""
    top = episodes.head(k)
    fig, axes = plt.subplots(len(top), 1, figsize=(9, 1.6 * len(top)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (_, ep) in zip(axes, top.iterrows(), strict=True):
        header = wfdb.rdheader(str(data_dir / ep["record"]))
        start = int(ep["start_s"] * header.fs)
        stop = min(start + 10 * header.fs, header.sig_len)
        rec = wfdb.rdrecord(str(data_dir / ep["record"]), channels=[0], sampfrom=start, sampto=stop)
        raw = np.nan_to_num(rec.p_signal[:, 0], nan=0.0)
        signal = sosfiltfilt(bandpass_sos(), resample_to_fs(raw, header.fs).astype(np.float64))
        t = np.arange(len(signal)) / FS
        ax.plot(t, signal, lw=0.6, color="#20415e")
        peaks = processing.xqrs_detect(signal, fs=FS, verbose=False)
        ax.plot(t[peaks], signal[peaks], "r.", ms=4)
        ax.set_yticks([])
        ax.set_ylabel(
            f"{ep['record']}\n{ep['start_s'] / 3600:.1f} h",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.set_title(
            f"AF probability {ep['mean_p_af']:.2f}  RR variation {ep['cv_rr']:.2f}  "
            f"{ep['beats']:.0f} beats  vetting {'passed' if ep['vet_pass'] else 'failed'}",
            fontsize=8,
            loc="left",
        )
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
