"""Generation of individual-specific 3D tissue geometry."""

from __future__ import annotations

import numpy as np

from .config import GeometryConfig
from .data import CanonicalAnatomy, GeometryTruth, IndividualGeometry
from .random import smooth_random_field


def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
    ax, ay, az = angles
    cx, cy, cz = np.cos([ax, ay, az])
    sx, sy, sz = np.sin([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


class GeometryModel:
    def __init__(self, canonical: CanonicalAnatomy):
        self.canonical = canonical

    def sample(
        self,
        individual_id: str,
        config: GeometryConfig,
        rng: np.random.Generator,
    ) -> IndividualGeometry:
        n_reference = len(self.canonical.coordinates)
        n_points = config.n_points or n_reference
        replace = n_points > n_reference
        source_indices = rng.choice(n_reference, size=n_points, replace=replace)
        # Sample biological identities from the canonical CCF support.  These
        # canonical coordinates must remain inside the selected CCF ROI because
        # Domain, cell-type probability, and expression fields are all evaluated
        # in canonical anatomical space.
        canonical_coordinates = self.canonical.coordinates[source_indices].copy()
        if self.canonical.voxel_resolution_um is not None and config.ccf_subvoxel_jitter:
            # Canonical CCF points are voxel centres.  Jitter by < half a voxel
            # so the point remains inside the very same annotated voxel.
            half_width = 0.49 * float(self.canonical.voxel_resolution_um)
            canonical_coordinates += rng.uniform(
                -half_width, half_width, size=canonical_coordinates.shape
            )
        canonical_normalized = self.canonical.normalize(canonical_coordinates)

        # Individual-specific coordinate jitter belongs to the *physical* tissue
        # realization, not the canonical CCF coordinate.  In v0.3.2 this jitter
        # was accidentally written back into canonical_coordinates, which could
        # move HY points across the atlas boundary and produce non-HY labels.
        physical_source_normalized = canonical_normalized.copy()
        if config.coordinate_jitter:
            physical_source_normalized += rng.normal(
                scale=config.coordinate_jitter, size=physical_source_normalized.shape
            )

        global_scale = float(np.exp(rng.normal(scale=config.scale_sd)))
        axis_scale = global_scale * np.exp(rng.normal(scale=config.anisotropy_sd, size=3))
        angles = np.deg2rad(rng.normal(scale=config.rotation_sd_degrees, size=3))
        rotation = _rotation_matrix(angles)
        affine = rotation @ np.diag(axis_scale)
        transformed = np.einsum("nd,ed->ne", physical_source_normalized, affine)

        # The smooth deformation field is anchored to the canonical anatomy.
        field = smooth_random_field(
            canonical_normalized,
            n_outputs=4,
            length_scale=config.warp_length_scale,
            n_features=config.random_features,
            rng=rng,
        )
        displacement = config.warp_amplitude * field[:, :3]
        radius = np.linalg.norm(canonical_normalized, axis=1, keepdims=True)
        radial_direction = canonical_normalized / np.maximum(radius, 1e-8)
        displacement += config.boundary_strength * field[:, 3:4] * radial_direction * radius
        transformed += displacement

        coordinates = self.canonical.denormalize(transformed)
        if not np.isfinite(coordinates).all():
            raise RuntimeError("geometry generation produced non-finite coordinates")
        return IndividualGeometry(
            individual_id=individual_id,
            canonical_coordinates=canonical_coordinates,
            coordinates=coordinates,
            normalized_coordinates=transformed,
            truth=GeometryTruth(
                source_indices=np.asarray(source_indices, dtype=np.int64),
                affine=affine,
                displacement=displacement,
                global_scale=global_scale,
            ),
        )
