"""Independent deterministic random streams for paired simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


GEOMETRY_STREAM = 11
DOMAIN_STREAM = 23
CELL_TYPE_STREAM = 37
EXPRESSION_STREAM = 53
SECTION_STREAM = 71
INTERVENTION_STREAM = 89


@dataclass(frozen=True)
class SeedTree:
    root_seed: int

    def rng(self, *path: int) -> np.random.Generator:
        if any(int(part) < 0 for part in path):
            raise ValueError("seed path components must be non-negative")
        sequence = np.random.SeedSequence([int(self.root_seed), *(int(p) for p in path)])
        return np.random.default_rng(sequence)


def smooth_random_field(
    coordinates: NDArray[np.float64],
    n_outputs: int,
    length_scale: float,
    n_features: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample a zero-mean unit-variance smooth field with random Fourier features."""
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("coordinates must have shape [n, 3]")
    if n_outputs < 1:
        return np.empty((x.shape[0], 0), dtype=float)
    omega = rng.normal(size=(3, n_features)) / length_scale
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    features = np.sqrt(2.0 / n_features) * np.cos(
        np.einsum("nd,df->nf", x, omega) + phase
    )
    coefficients = rng.normal(size=(n_features, n_outputs))
    field = np.einsum("nf,fo->no", features, coefficients)
    field -= field.mean(axis=0, keepdims=True)
    scale = field.std(axis=0, keepdims=True)
    return field / np.where(scale > 1e-8, scale, 1.0)
