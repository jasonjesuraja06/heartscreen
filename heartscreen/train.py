"""Fold training: AdamW with cosine schedule, class-weighted loss, jit-compiled steps."""

from __future__ import annotations

import csv
import dataclasses
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import serialization
from flax.training.train_state import TrainState

from heartscreen.models import ECGResNet
from heartscreen.preprocessing import FS, fixed_window, make_batches


@dataclasses.dataclass
class Config:
    seed: int = 42
    data_dir: str = "data/cinc2017"
    cache: str = "data/cache/cinc2017.npz"
    out_dir: str = "results/cv"
    window_seconds: int = 30
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    widths: tuple[int, ...] = (32, 64, 96, 128)
    blocks_per_stage: int = 2
    kernel: int = 7
    folds: int = 5
    max_eval_crops: int = 3
    limit_records: int | None = None

    @property
    def window(self) -> int:
        return self.window_seconds * FS


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw["widths"] = tuple(raw.get("widths", Config.widths))
    return Config(**raw)


def class_weights(labels: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Inverse-frequency weights normalized to mean 1 over the given labels."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    return (len(labels) / (num_classes * np.maximum(counts, 1))).astype(np.float32)


def build_model(cfg: Config) -> ECGResNet:
    return ECGResNet(tuple(cfg.widths), cfg.blocks_per_stage, cfg.kernel)


def create_state(cfg: Config, model: ECGResNet, steps_per_epoch: int) -> TrainState:
    params = model.init(
        jax.random.key(cfg.seed), jnp.zeros((1, cfg.window, 1)), jnp.ones((1, cfg.window))
    )["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_epochs * steps_per_epoch,
        decay_steps=cfg.epochs * steps_per_epoch,
    )
    tx = optax.adamw(schedule, weight_decay=cfg.weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_steps(apply_fn, weights: np.ndarray):
    weights = jnp.asarray(weights)

    @jax.jit
    def train_step(state, x, mask, y):
        def loss_fn(params):
            logits = apply_fn({"params": params}, x, mask)
            logp = jax.nn.log_softmax(logits)
            ce = -jnp.take_along_axis(logp, y[:, None], axis=1)[:, 0]
            return jnp.mean(ce * weights[y])

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_step(params, x, mask):
        return apply_fn({"params": params}, x, mask)

    return train_step, eval_step


def predict_logits(eval_step, params, signals: list[np.ndarray], cfg: Config) -> np.ndarray:
    """Per-record logits averaged over up to max_eval_crops evenly spaced windows."""
    length = cfg.window
    crops, masks, owners = [], [], []
    for i, sig in enumerate(signals):
        if len(sig) <= length:
            w, m = fixed_window(sig, length)
            crops.append(w)
            masks.append(m)
            owners.append(i)
        else:
            num = min(cfg.max_eval_crops, int(np.ceil(len(sig) / length)))
            starts = np.unique(np.linspace(0, len(sig) - length, num).round().astype(int))
            for start in starts:
                crops.append(sig[start : start + length])
                masks.append(np.ones(length, np.float32))
                owners.append(i)

    total = np.zeros((len(signals), 4), np.float64)
    counts = np.zeros(len(signals), np.int64)
    bs = cfg.batch_size
    for lo in range(0, len(crops), bs):
        xb = np.stack(crops[lo : lo + bs])
        mb = np.stack(masks[lo : lo + bs])
        n_real = len(xb)
        if n_real < bs:
            # Pad to the training batch shape so jit does not recompile.
            xb = np.concatenate([xb, np.zeros((bs - n_real, length), np.float32)])
            mb = np.concatenate([mb, np.ones((bs - n_real, length), np.float32)])
        out = np.asarray(eval_step(params, xb[..., None], mb))[:n_real]
        for j, owner in enumerate(owners[lo : lo + n_real]):
            total[owner] += out[j]
            counts[owner] += 1
    return (total / counts[:, None]).astype(np.float32)


def train_fold(
    cfg: Config,
    train_signals: list[np.ndarray],
    y_train: np.ndarray,
    val_signals: list[np.ndarray],
    y_val: np.ndarray,
    out_dir: str | Path,
    metric,
) -> dict:
    """Train one fold and return final params, val logits, and per-epoch history."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg)
    steps_per_epoch = len(train_signals) // cfg.batch_size
    state = create_state(cfg, model, steps_per_epoch)
    train_step, eval_step = make_steps(model.apply, class_weights(y_train))
    rng = np.random.default_rng(cfg.seed)

    history = []
    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_f1", "seconds"])
        for epoch in range(cfg.epochs):
            start = time.perf_counter()
            losses = []
            batches = make_batches(
                train_signals,
                y_train,
                cfg.window,
                cfg.batch_size,
                rng,
                augment=True,
                drop_last=True,
            )
            for x, mask, y in batches:
                state, loss = train_step(state, x, mask, y)
                losses.append(loss)
            train_loss = float(np.mean(jax.device_get(losses)))
            val_logits = predict_logits(eval_step, state.params, val_signals, cfg)
            val_f1 = metric(y_val, val_logits.argmax(1))
            seconds = time.perf_counter() - start
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1, "seconds": seconds}
            )
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_f1:.4f}", f"{seconds:.1f}"])
            f.flush()

    (out_dir / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    val_logits = predict_logits(eval_step, state.params, val_signals, cfg)
    return {"params": state.params, "val_logits": val_logits, "history": history}


def load_params(cfg: Config, path: str | Path):
    """Restore saved fold parameters into the model's parameter structure."""
    model = build_model(cfg)
    template = model.init(
        jax.random.key(0), jnp.zeros((1, cfg.window, 1)), jnp.ones((1, cfg.window))
    )["params"]
    return serialization.from_bytes(template, Path(path).read_bytes())
