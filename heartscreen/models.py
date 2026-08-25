"""Residual 1D CNN for fixed-length ECG windows with padding masks."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class ResBlock(nn.Module):
    features: int
    kernel: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.features, (self.kernel,), strides=(self.stride,))(x)
        y = nn.GroupNorm(num_groups=8)(y)
        y = nn.relu(y)
        y = nn.Conv(self.features, (self.kernel,))(y)
        y = nn.GroupNorm(num_groups=8)(y)
        if residual.shape != y.shape:
            residual = nn.Conv(self.features, (1,), strides=(self.stride,))(residual)
        return nn.relu(y + residual)


class ECGResNet(nn.Module):
    """Stacked residual stages over a strided stem, masked global average pool, 4-way head.

    GroupNorm keeps statistics per sample, so right-padded batches need masking
    only at the pooling step. Padding is always on the right and SAME convs use
    ceil division, so a sample with v valid input steps has ceil(v / downsample)
    valid output steps.
    """

    widths: tuple[int, ...] = (32, 64, 96, 128)
    blocks_per_stage: int = 2
    kernel: int = 7
    num_classes: int = 4

    @property
    def downsample(self) -> int:
        return 4 * 2 ** (len(self.widths) - 1)

    @nn.compact
    def __call__(self, x, mask):
        x = nn.Conv(self.widths[0], (15,), strides=(2,))(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3,), strides=(2,), padding="SAME")
        for i, width in enumerate(self.widths):
            for j in range(self.blocks_per_stage):
                stride = 2 if i > 0 and j == 0 else 1
                x = ResBlock(width, self.kernel, stride)(x)

        valid = jnp.ceil(jnp.sum(mask, axis=1) / self.downsample)
        pooled_mask = (jnp.arange(x.shape[1])[None, :] < valid[:, None]).astype(x.dtype)
        counts = jnp.maximum(jnp.sum(pooled_mask, axis=1, keepdims=True), 1.0)
        x = jnp.sum(x * pooled_mask[..., None], axis=1) / counts
        return nn.Dense(self.num_classes)(x)
