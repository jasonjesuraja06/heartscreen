"""Residual 1D CNN for fixed-length ECG windows with padding masks."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def length_mask(valid: jnp.ndarray, length: int, dtype) -> jnp.ndarray:
    return (jnp.arange(length)[None, :] < valid[:, None]).astype(dtype)


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose statistics cover only valid (unpadded) positions.

    Plain GroupNorm reduces over the time axis, so right-padding would fold
    zeros into each sample's mean and variance and rescale short records by
    their pad fraction. Outputs are re-zeroed at padded positions so padding
    stays exactly zero into the next layer.
    """

    num_groups: int = 8
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x, mask):
        b, t, c = x.shape
        grouped = x.reshape(b, t, self.num_groups, c // self.num_groups)
        m = mask[:, :, None, None]
        count = jnp.maximum(jnp.sum(m, axis=1, keepdims=True) * (c // self.num_groups), 1.0)
        mean = jnp.sum(grouped * m, axis=(1, 3), keepdims=True) / count
        var = jnp.sum(jnp.square(grouped - mean) * m, axis=(1, 3), keepdims=True) / count
        grouped = (grouped - mean) / jnp.sqrt(var + self.eps)
        scale = self.param("scale", nn.initializers.ones, (c,))
        bias = self.param("bias", nn.initializers.zeros, (c,))
        return (grouped.reshape(b, t, c) * scale + bias) * mask[..., None]


class ResBlock(nn.Module):
    features: int
    kernel: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, valid):
        residual = x
        y = nn.Conv(self.features, (self.kernel,), strides=(self.stride,))(x)
        valid = jnp.ceil(valid / self.stride)
        mask = length_mask(valid, y.shape[1], y.dtype)
        y = MaskedGroupNorm()(y, mask)
        y = nn.relu(y)
        y = nn.Conv(self.features, (self.kernel,))(y)
        y = MaskedGroupNorm()(y, mask)
        if residual.shape != y.shape:
            residual = nn.Conv(self.features, (1,), strides=(self.stride,))(residual)
        return nn.relu(y + residual) * mask[..., None], valid


class ECGResNet(nn.Module):
    """Stacked residual stages over a strided stem, masked global average pool, 4-way head.

    Valid lengths are tracked through every stride-2 op (SAME convolutions use
    ceil division, and padding is always on the right), so each normalization
    and the final pool see exactly the unpadded positions.
    """

    widths: tuple[int, ...] = (32, 64, 96, 128)
    blocks_per_stage: int = 2
    kernel: int = 7
    num_classes: int = 4

    @nn.compact
    def __call__(self, x, mask):
        valid = jnp.sum(mask, axis=1)
        x = nn.Conv(self.widths[0], (15,), strides=(2,))(x)
        valid = jnp.ceil(valid / 2)
        x = MaskedGroupNorm()(x, length_mask(valid, x.shape[1], x.dtype))
        x = nn.relu(x)
        x = nn.max_pool(x, (3,), strides=(2,), padding="SAME")
        valid = jnp.ceil(valid / 2)
        for i, width in enumerate(self.widths):
            for j in range(self.blocks_per_stage):
                stride = 2 if i > 0 and j == 0 else 1
                x, valid = ResBlock(width, self.kernel, stride)(x, valid)

        pooled_mask = length_mask(valid, x.shape[1], x.dtype)
        counts = jnp.maximum(jnp.sum(pooled_mask, axis=1, keepdims=True), 1.0)
        x = jnp.sum(x * pooled_mask[..., None], axis=1) / counts
        return nn.Dense(self.num_classes)(x)
