import numpy as np

from heartscreen.evaluate import challenge_f1, confusion
from heartscreen.train import Config, class_weights, train_fold


def test_class_weights():
    labels = np.array([0] * 60 + [1] * 10 + [2] * 25 + [3] * 5)
    weights = class_weights(labels)
    expected = 100 / (4 * np.array([60, 10, 25, 5]))
    np.testing.assert_allclose(weights, expected, rtol=1e-6)


def test_challenge_f1_hand_computed():
    y_true = np.array([0, 0, 1, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3])
    score, per_class = challenge_f1(y_true, y_pred)
    assert per_class["N"] == 2 * 1 / 3
    assert per_class["A"] == 2 * 2 / 5
    assert per_class["O"] == 1.0
    assert np.isclose(score, (2 / 3 + 4 / 5 + 1) / 3)
    assert per_class["~"] == 1.0


def test_confusion_matrix():
    matrix = confusion(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 3]))
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 1] == 1 and matrix[2, 3] == 1
    assert matrix.sum() == 4


def test_train_fold_learns_synthetic(tmp_path):
    # Classes are separable by amplitude, so two epochs should be enough to
    # beat chance and drive the loss down.
    rng = np.random.default_rng(0)
    cfg = Config(
        window_seconds=1,
        batch_size=8,
        epochs=2,
        warmup_epochs=1,
        widths=(8, 16),
        blocks_per_stage=1,
        lr=1e-2,
        max_eval_crops=1,
    )
    amplitudes = [0.1, 1.0, 3.0, 8.0]

    def make(n):
        labels = rng.integers(0, 4, n).astype(np.int32)
        signals = [
            (amplitudes[y] * np.sin(np.linspace(0, 20, 300))).astype(np.float32) for y in labels
        ]
        return signals, labels

    train_signals, y_train = make(64)
    val_signals, y_val = make(32)
    metric = lambda yt, yp: challenge_f1(yt, yp)[0]  # noqa: E731
    result = train_fold(cfg, train_signals, y_train, val_signals, y_val, tmp_path, metric)
    assert (tmp_path / "log.csv").exists()
    assert (tmp_path / "params.msgpack").exists()
    assert result["val_logits"].shape == (32, 4)
    assert len(result["history"]) == 2
    assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]
