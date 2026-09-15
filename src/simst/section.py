"""Idealized physical sectioning with correlated pose variation and no assay noise."""

from __future__ import annotations

import numpy as np

from .config import SectionConfig
from .data import BiologicalTissue, IdealSection


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(normal, candidate))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    first = candidate - np.dot(candidate, normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second


class SectionSampler:
    def sample(
        self,
        tissue: BiologicalTissue,
        config: SectionConfig,
        rng: np.random.Generator,
    ) -> tuple[IdealSection, ...]:
        coordinates = tissue.geometry.coordinates
        low, high = np.quantile(coordinates[:, 2], [0.03, 0.97])
        nominal_z = np.linspace(low, high, config.n_sections)
        if config.target_counts is not None and len(config.target_counts) != config.n_sections:
            raise ValueError("target_counts must have exactly n_sections entries")
        rho = config.adjacent_correlation
        innovation_scale = np.sqrt(max(0.0, 1.0 - rho * rho))
        rotation_state = np.zeros(2, dtype=float)
        translation_state = np.zeros(3, dtype=float)
        sections: list[IdealSection] = []

        for index, z in enumerate(nominal_z):
            rotation_state = rho * rotation_state + innovation_scale * rng.normal(size=2)
            translation_state = rho * translation_state + innovation_scale * rng.normal(size=3)
            angles = np.deg2rad(config.rotation_sd_degrees) * rotation_state
            normal = np.array([np.tan(angles[0]), np.tan(angles[1]), 1.0], dtype=float)
            normal /= np.linalg.norm(normal)
            origin = np.array([np.median(coordinates[:, 0]), np.median(coordinates[:, 1]), z])
            origin += config.translation_sd * translation_state
            signed_distance = np.einsum("nd,d->n", coordinates - origin, normal)
            if config.target_counts is None:
                source_index = np.flatnonzero(np.abs(signed_distance) <= config.thickness / 2.0)
            else:
                target = int(config.target_counts[index])
                if target < config.min_points:
                    raise ValueError("Each target section count must be >= min_points")
                if target > len(coordinates):
                    raise ValueError("A target section count exceeds the 3D tissue cell count")
                # Exact-size MVP calibration: select the cells closest to the plane.
                source_index = np.argpartition(np.abs(signed_distance), target - 1)[:target]
                source_index = np.asarray(source_index, dtype=np.int64)
            if len(source_index) < config.min_points:
                raise ValueError(
                    f"{index}: section contains {len(source_index)} points; "
                    f"minimum is {config.min_points}"
                )
            first, second = _plane_basis(normal)
            relative = coordinates[source_index] - origin
            coordinates_2d = np.column_stack(
                (
                    np.einsum("nd,d->n", relative, first),
                    np.einsum("nd,d->n", relative, second),
                )
            )
            sections.append(
                IdealSection(
                    section_id=f"section_{index:03d}",
                    source_indices=np.asarray(source_index, dtype=np.int64),
                    coordinates_2d=coordinates_2d,
                    plane_origin=origin,
                    plane_normal=normal,
                    expression=tissue.expression[source_index],
                    domain_index=tissue.domain_index[source_index],
                    cell_type_index=tissue.cell_type_index[source_index],
                )
            )
        return tuple(sections)
