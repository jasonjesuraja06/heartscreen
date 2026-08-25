import jax
import jax.numpy as jnp
import numpy as np

from heartscreen.models import ECGResNet


def init_params(model, seed=0, length=1024):
    return model.init(jax.random.key(seed), jnp.zeros((1, length, 1)), jnp.ones((1, length)))[
        "params"
    ]


def test_output_shape():
    model = ECGResNet(widths=(8, 16), blocks_per_stage=1)
    params = init_params(model)
    x = jnp.zeros((3, 1024, 1))
    mask = jnp.ones((3, 1024))
    assert model.apply({"params": params}, x, mask).shape == (3, 4)


def test_parameter_count_under_budget():
    model = ECGResNet()
    params = init_params(model, length=9000)
    count = sum(p.size for p in jax.tree.leaves(params))
    assert 100_000 < count < 1_000_000


def test_seed_determinism():
    model = ECGResNet(widths=(8, 16), blocks_per_stage=1)
    a = init_params(model, seed=1)
    b = init_params(model, seed=1)
    c = init_params(model, seed=2)
    flat_a, flat_b = jax.tree.leaves(a), jax.tree.leaves(b)
    for pa, pb in zip(flat_a, flat_b, strict=True):
        np.testing.assert_array_equal(pa, pb)
    assert any(
        not np.array_equal(pa, pc) for pa, pc in zip(flat_a, jax.tree.leaves(c), strict=True)
    )


def test_padding_in_batch_does_not_change_logits():
    # GroupNorm plus masked pooling: a sample's logits must not depend on how
    # much padding its batch neighbors carry.
    model = ECGResNet(widths=(8, 16), blocks_per_stage=1)
    params = init_params(model)
    rng = np.random.default_rng(0)
    full = rng.normal(size=(1024,)).astype(np.float32)
    short = np.zeros(1024, np.float32)
    short[:400] = rng.normal(size=400)
    short_mask = np.zeros(1024, np.float32)
    short_mask[:400] = 1

    alone = model.apply({"params": params}, full[None, :, None], np.ones((1, 1024), np.float32))
    batched = model.apply(
        {"params": params},
        np.stack([full, short])[..., None],
        np.stack([np.ones(1024, np.float32), short_mask]),
    )
    np.testing.assert_allclose(alone[0], batched[0], atol=1e-5)
