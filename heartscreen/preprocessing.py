"""Signal conditioning and windowing for 300 Hz single-lead ECG."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

FS = 300
BAND = (0.5, 40.0)


@lru_cache
def bandpass_sos(fs: int = FS, low: float = BAND[0], high: float = BAND[1], order: int = 4):
    return butter(order, (low, high), btype="bandpass", fs=fs, output="sos")


def condition(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Zero-phase bandpass then per-record z-score; returns float32."""
    filtered = sosfiltfilt(bandpass_sos(fs), signal.astype(np.float64))
    std = filtered.std()
    if std < 1e-8:
        std = 1e-8
    return ((filtered - filtered.mean()) / std).astype(np.float32)


def resample_to_fs(signal: np.ndarray, fs_in: float, fs_out: int = FS) -> np.ndarray:
    frac = Fraction(fs_out, int(round(fs_in))).limit_denominator(1000)
    return resample_poly(signal, frac.numerator, frac.denominator).astype(np.float32)


def fixed_window(
    signal: np.ndarray, length: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Crop (random with rng, else centered) or right-pad to length; returns (window, mask)."""
    n = len(signal)
    if n >= length:
        start = (n - length) // 2 if rng is None else int(rng.integers(0, n - length + 1))
        return signal[start : start + length], np.ones(length, np.float32)
    out = np.zeros(length, np.float32)
    out[:n] = signal
    mask = np.zeros(length, np.float32)
    mask[:n] = 1.0
    return out, mask


def make_batches(
    signals: list[np.ndarray],
    labels: np.ndarray,
    length: int,
    batch_size: int,
    rng: np.random.Generator | None = None,
    augment: bool = False,
    drop_last: bool = False,
):
    """Yield (x[B, L, 1], mask[B, L], y[B]) batches; shuffles and augments when rng is given."""
    order = np.arange(len(signals))
    if rng is not None:
        rng.shuffle(order)
    for lo in range(0, len(order), batch_size):
        idx = order[lo : lo + batch_size]
        if drop_last and len(idx) < batch_size:
            return
        xs, masks = zip(*(fixed_window(signals[i], length, rng) for i in idx), strict=True)
        x = np.stack(xs)
        if augment and rng is not None:
            # Amplitude scale and polarity flip; the AliveCor device records
            # either polarity depending on how it is held.
            scale = rng.uniform(0.8, 1.2, size=(len(idx), 1)).astype(np.float32)
            flip = rng.choice([-1.0, 1.0], size=(len(idx), 1)).astype(np.float32)
            x = x * scale * flip
        yield x[..., None], np.stack(masks), labels[idx]
