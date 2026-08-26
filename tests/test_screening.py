import numpy as np

from heartscreen.preprocessing import FS
from heartscreen.screening import (
    Episode,
    af_fraction,
    find_episodes,
    rhythm_intervals,
    rr_metrics,
    vet,
    window_sqi,
)


class FakeAnnotation:
    def __init__(self, samples, aux_notes):
        self.sample = np.array(samples)
        self.aux_note = aux_notes


def test_rhythm_intervals():
    ann = FakeAnnotation([0, 500, 500, 900], ["(N", "N", "(AFIB", "(N"])
    intervals = rhythm_intervals(ann, 1200)
    assert intervals == [(0, 500, "(N"), (500, 900, "(AFIB"), (900, 1200, "(N")]


def test_af_fraction():
    intervals = [(0, 100, "(N"), (100, 300, "(AFIB"), (300, 400, "(N")]
    assert af_fraction(intervals, 0, 400) == 0.5
    assert af_fraction(intervals, 100, 300) == 1.0
    assert af_fraction(intervals, 0, 100) == 0.0
    assert af_fraction(intervals, 50, 150) == 0.5


def test_find_episodes_merges_consecutive_windows():
    starts = np.arange(5) * 100
    probs = np.zeros((5, 4), np.float32)
    probs[:, 1] = [0.9, 0.8, 0.2, 0.7, 0.6]
    episodes = find_episodes("rec", starts, probs, window=250, threshold=0.5)
    assert len(episodes) == 2
    first = episodes[0]
    assert isinstance(first, Episode)
    assert first.n_windows == 2
    assert first.start_s == 0
    # The episode extends to the end of the last hot window, not its start.
    assert first.end_s == (100 + 250) / FS
    assert np.isclose(first.mean_p_af, 0.85)
    assert episodes[1].n_windows == 2


def test_sliding_probs_short_recording_yields_no_windows():
    from heartscreen.screening import sliding_probs
    from heartscreen.train import Config

    cfg = Config(window_seconds=1, batch_size=4)
    starts, probs = sliding_probs(None, None, np.zeros(200, np.float32), cfg, stride=100)
    assert len(starts) == 0
    assert probs.shape == (0, 4)


def test_window_sqi_flags_flatline():
    good = np.sin(np.arange(3000) / 10).astype(np.float32)
    flat = np.zeros(3000, np.float32)
    assert window_sqi(flat)["flatline_frac"] > 0.9
    assert window_sqi(good)["flatline_frac"] < 0.1


def test_rr_metrics_on_synthetic_ecg():
    # Impulse train at 1 Hz with alternating intervals gives detectable peaks
    # and a positive irregularity index.
    window = np.zeros(30 * FS)
    t = 0.0
    intervals = [0.7, 1.1] * 30
    i = 0
    while t < 29:
        center = int(t * FS)
        window[center : center + 12] = np.hanning(12) * 2
        t += intervals[i % len(intervals)]
        i += 1
    metrics = rr_metrics(window.astype(np.float32))
    assert metrics["beats"] > 20
    assert metrics["cv_rr"] > 0.1


def test_vet_rejects_bad_signal():
    good_sqi = {"flatline_frac": 0.01, "qrs_band_ratio": 0.6}
    irregular = {"beats": 35.0, "cv_rr": 0.25, "rmssd_ms": 120.0}
    assert vet(good_sqi, irregular, 30)
    assert not vet({"flatline_frac": 0.5, "qrs_band_ratio": 0.6}, irregular, 30)
    assert not vet(good_sqi, {"beats": 35.0, "cv_rr": 0.03, "rmssd_ms": 10.0}, 30)
    assert not vet(good_sqi, {"beats": 2.0, "cv_rr": np.nan, "rmssd_ms": np.nan}, 30)
