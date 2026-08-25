"""Measure jit-compiled versus eager train step time on one full-size batch."""

import time

import jax
import numpy as np

from heartscreen.train import Config, class_weights, create_state, make_steps


def time_step(step, state, batch, repeats):
    state, loss = step(state, *batch)
    jax.block_until_ready(loss)
    start = time.perf_counter()
    for _ in range(repeats):
        state, loss = step(state, *batch)
    jax.block_until_ready(loss)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    x = rng.normal(size=(cfg.batch_size, cfg.window, 1)).astype(np.float32)
    mask = np.ones((cfg.batch_size, cfg.window), np.float32)
    y = rng.integers(0, 4, cfg.batch_size).astype(np.int32)
    weights = class_weights(y)

    from heartscreen.train import build_model

    model = build_model(cfg)
    state = create_state(cfg, model, steps_per_epoch=100)
    jit_step, _ = make_steps(model.apply, weights)

    jit_time = time_step(jit_step, state, (x, mask, y), repeats=20)
    with jax.disable_jit():
        eager_time = time_step(jit_step, state, (x, mask, y), repeats=3)

    per_record = jit_time / cfg.batch_size
    print(f"jit step: {jit_time * 1000:.1f} ms  eager step: {eager_time * 1000:.1f} ms")
    print(f"speedup: {eager_time / jit_time:.1f}x")
    print(f"training throughput: {cfg.batch_size / jit_time:.0f} windows/s "
          f"({per_record * 1000:.2f} ms per 30 s window)")


if __name__ == "__main__":
    main()
