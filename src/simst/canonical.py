"""Canonical anatomical context constructed from reference 3D locations."""

from __future__ import annotations

import numpy as np

from .data import CanonicalAnatomy, ReferenceData


def build_canonical_anatomy(reference: ReferenceData) -> CanonicalAnatomy:
    coordinates = reference.coordinates
    center = np.median(coordinates, axis=0)
    low, high = np.quantile(coordinates, [0.01, 0.99], axis=0)
    scale = high - low
    if (scale == 0).any():
        raise ValueError("reference coordinates must vary along all three axes")
    normalized = (coordinates - center) / scale
    return CanonicalAnatomy(
        coordinates=coordinates.copy(),
        normalized_coordinates=normalized,
        center=center,
        scale=scale,
        domain_names=np.unique(reference.domain_labels),
        cell_type_names=np.unique(reference.cell_type_labels),
        gene_names=reference.gene_names.copy(),
    )
