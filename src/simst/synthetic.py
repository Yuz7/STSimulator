"""Synthetic reference generator used for examples and smoke testing."""

from __future__ import annotations

import numpy as np

from .data import ReferenceData


def make_reference(
    n_locations: int = 1200,
    n_genes: int = 24,
    seed: int = 7,
) -> ReferenceData:
    if n_locations < 20 or n_genes < 2:
        raise ValueError("n_locations must be >= 20 and n_genes >= 2")
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    n_accepted = 0
    while n_accepted < n_locations:
        candidates = rng.uniform(-1.0, 1.0, size=(n_locations, 3))
        inside = candidates[
            np.linalg.norm(candidates / np.array([1.0, 0.8, 0.65]), axis=1) <= 1
        ]
        accepted.append(inside)
        n_accepted += len(inside)
    coordinates = np.vstack(accepted)[:n_locations]
    domain_index = np.where(coordinates[:, 2] > 0.2, 2, np.where(coordinates[:, 0] > 0, 1, 0))
    domain_names = np.array(["outer", "right", "upper"])
    domain_labels = domain_names[domain_index]

    cell_logits = np.column_stack(
        [
            1.3 * (domain_index == 0) - 0.5 * coordinates[:, 1],
            1.2 * (domain_index == 1) + 0.8 * coordinates[:, 2],
            1.4 * (domain_index == 2) + 0.5 * coordinates[:, 0],
            0.2 + 0.6 * np.sin(3 * coordinates[:, 1]),
        ]
    )
    cell_logits += rng.normal(scale=0.35, size=cell_logits.shape)
    cell_index = np.argmax(cell_logits, axis=1)
    cell_names = np.array(["astrocyte", "excitatory", "inhibitory", "ependymal"])
    cell_type_labels = cell_names[cell_index]

    gene_names = np.asarray([f"gene_{index:03d}" for index in range(n_genes)])
    baseline = rng.normal(1.2, 0.35, size=n_genes)
    domain_effect = rng.normal(scale=0.4, size=(len(domain_names), n_genes))
    cell_effect = rng.normal(scale=0.55, size=(len(cell_names), n_genes))
    spatial_loading = rng.normal(scale=0.25, size=(3, n_genes))
    log_mean = (
        baseline
        + domain_effect[domain_index]
        + cell_effect[cell_index]
        + np.einsum("nd,dg->ng", coordinates, spatial_loading)
    )
    expression = np.maximum(np.expm1(log_mean + rng.normal(scale=0.2, size=log_mean.shape)), 0)
    return ReferenceData(
        coordinates=coordinates,
        domain_labels=domain_labels,
        cell_type_labels=cell_type_labels,
        expression=expression,
        gene_names=gene_names,
        metadata={"source": "simst.synthetic"},
    )
