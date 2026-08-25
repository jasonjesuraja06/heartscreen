import numpy as np

from heartscreen.preprocessing import FS, condition, fixed_window, make_batches, resample_to_fs


def band_power(x, freq, fs=FS):
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return spectrum[np.argmin(np.abs(freqs - freq))]


def test_bandpass_attenuates_out_of_band():
    t = np.arange(30 * FS) / FS
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 90 * t)
    out = condition(signal)
    assert band_power(out, 10) / band_power(out, 0.1) > 50
    assert band_power(out, 10) / band_power(out, 90) > 50


def test_condition_normalizes():
    rng = np.random.default_rng(0)
    out = condition(rng.normal(2.0, 5.0, 6000))
    assert abs(out.mean()) < 0.05
    assert abs(out.std() - 1.0) < 0.05
    assert out.dtype == np.float32


def test_condition_constant_signal():
    out = condition(np.ones(3000))
    assert np.all(np.isfinite(out))


def test_fixed_window_pads_short():
    window, mask = fixed_window(np.ones(100, np.float32), 250)
    assert window.shape == (250,)
    assert np.all(window[:100] == 1) and np.all(window[100:] == 0)
    assert mask.sum() == 100


def test_fixed_window_crops_long():
    signal = np.arange(500, dtype=np.float32)
    window, mask = fixed_window(signal, 200)
    assert window[0] == 150  # centered
    assert mask.sum() == 200
    rng = np.random.default_rng(0)
    starts = {fixed_window(signal, 200, rng)[0][0] for _ in range(20)}
    assert len(starts) > 1


def test_make_batches_shapes_and_determinism():
    rng = np.random.default_rng(3)
    signals = [rng.normal(size=rng.integers(80, 300)).astype(np.float32) for _ in range(10)]
    labels = np.arange(10, dtype=np.int32) % 4

    batches = list(make_batches(signals, labels, 128, 4, np.random.default_rng(7), augment=True))
    assert [b[0].shape for b in batches] == [(4, 128, 1), (4, 128, 1), (2, 128, 1)]
    assert all(b[1].shape == b[0].shape[:2] for b in batches)

    again = list(make_batches(signals, labels, 128, 4, np.random.default_rng(7), augment=True))
    for (x1, m1, y1), (x2, m2, y2) in zip(batches, again, strict=True):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(y1, y2)

    dropped = list(make_batches(signals, labels, 128, 4, drop_last=True))
    assert len(dropped) == 2


def test_resample_preserves_frequency():
    fs_in = 128
    t = np.arange(fs_in * 10) / fs_in
    signal = np.sin(2 * np.pi * 5 * t)
    out = resample_to_fs(signal, fs_in)
    assert len(out) == 10 * FS
    peak = np.argmax(np.abs(np.fft.rfft(out)))
    freq = np.fft.rfftfreq(len(out), 1 / FS)[peak]
    assert abs(freq - 5.0) < 0.1
