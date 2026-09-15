"""Core data contracts for reference, latent tissue, and ideal sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_str_array(values: NDArray | list[str], name: str) -> NDArray[np.str_]:
    arr = np.asarray(values, dtype=str)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


@dataclass(frozen=True)
class ReferenceData:
    """Training ST data.

    ``coordinates`` are the original/experimental 3D coordinates when available.
    ``canonical_coordinates`` are the coordinates after registration to the CCFv3
    atlas.  The CCF-aware training modules use ``canonical_coordinates``.  For
    legacy/synthetic examples, when ``canonical_coordinates`` is omitted,
    ``coordinates`` are used as the canonical coordinates.

    ``domain_labels`` may contain placeholders on input.  A CCFv3DomainModule
    replaces them deterministically during ``SimST.fit`` before the cell-type and
    expression modules are trained.
    """

    coordinates: FloatArray
    domain_labels: NDArray[np.str_]
    cell_type_labels: NDArray[np.str_]
    expression: FloatArray
    gene_names: NDArray[np.str_]
    canonical_coordinates: FloatArray | None = None
    library_size: FloatArray | None = None
    section_id: NDArray[np.str_] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        expression = np.asarray(self.expression, dtype=float)
        domains = _as_str_array(self.domain_labels, "domain_labels")
        cell_types = _as_str_array(self.cell_type_labels, "cell_type_labels")
        genes = _as_str_array(self.gene_names, "gene_names")
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [n, 3]")
        n = coordinates.shape[0]
        if n < 2:
            raise ValueError("reference requires at least two locations")
        if expression.ndim != 2 or expression.shape != (n, len(genes)):
            raise ValueError("expression must have shape [n_locations, n_genes]")
        if len(domains) != n or len(cell_types) != n:
            raise ValueError("labels must have one value per location")
        if len(genes) < 1 or len(set(genes.tolist())) != len(genes):
            raise ValueError("gene_names must be non-empty and unique")
        if not np.isfinite(coordinates).all() or not np.isfinite(expression).all():
            raise ValueError("coordinates and expression must be finite")
        if (expression < 0).any():
            raise ValueError("expression must be non-negative")

        canonical = self.canonical_coordinates
        if canonical is not None:
            canonical = np.asarray(canonical, dtype=float)
            if canonical.shape != (n, 3) or not np.isfinite(canonical).all():
                raise ValueError("canonical_coordinates must have shape [n, 3] and be finite")

        library_size = self.library_size
        if library_size is not None:
            library_size = np.asarray(library_size, dtype=float)
            if library_size.shape != (n,):
                raise ValueError("library_size must have shape [n_locations]")
            if not np.isfinite(library_size).all() or (library_size <= 0).any():
                raise ValueError("library_size must be finite and strictly positive")

        section_id = self.section_id
        if section_id is not None:
            section_id = _as_str_array(section_id, "section_id")
            if len(section_id) != n:
                raise ValueError("section_id must have one value per location")

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "domain_labels", domains)
        object.__setattr__(self, "cell_type_labels", cell_types)
        object.__setattr__(self, "gene_names", genes)
        object.__setattr__(self, "canonical_coordinates", canonical)
        object.__setattr__(self, "library_size", library_size)
        object.__setattr__(self, "section_id", section_id)

    @property
    def ccf_coordinates(self) -> FloatArray:
        if self.canonical_coordinates is not None:
            return self.canonical_coordinates
        return self.coordinates

    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "coordinates": self.coordinates,
            "domain_labels": self.domain_labels,
            "cell_type_labels": self.cell_type_labels,
            "expression": self.expression,
            "gene_names": self.gene_names,
        }
        if self.canonical_coordinates is not None:
            payload["canonical_coordinates"] = self.canonical_coordinates
        if self.library_size is not None:
            payload["library_size"] = self.library_size
        if self.section_id is not None:
            payload["section_id"] = self.section_id
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceData":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                coordinates=data["coordinates"],
                domain_labels=data["domain_labels"],
                cell_type_labels=data["cell_type_labels"],
                expression=data["expression"],
                gene_names=data["gene_names"],
                canonical_coordinates=(
                    data["canonical_coordinates"] if "canonical_coordinates" in data else None
                ),
                library_size=data["library_size"] if "library_size" in data else None,
                section_id=data["section_id"] if "section_id" in data else None,
            )


@dataclass(frozen=True)
class CanonicalAnatomy:
    coordinates: FloatArray
    normalized_coordinates: FloatArray
    center: FloatArray
    scale: FloatArray
    domain_names: NDArray[np.str_]
    cell_type_names: NDArray[np.str_]
    gene_names: NDArray[np.str_]
    voxel_resolution_um: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalize(self, coordinates: FloatArray) -> FloatArray:
        return (np.asarray(coordinates, dtype=float) - self.center) / self.scale

    def denormalize(self, coordinates: FloatArray) -> FloatArray:
        return np.asarray(coordinates, dtype=float) * self.scale + self.center


@dataclass(frozen=True)
class GeometryTruth:
    source_indices: IntArray
    affine: FloatArray
    displacement: FloatArray
    global_scale: float


@dataclass(frozen=True)
class IndividualGeometry:
    individual_id: str
    canonical_coordinates: FloatArray
    coordinates: FloatArray
    normalized_coordinates: FloatArray
    truth: GeometryTruth

    def __post_init__(self) -> None:
        n = len(self.coordinates)
        if self.coordinates.shape != (n, 3):
            raise ValueError("individual coordinates must have shape [n, 3]")
        if self.canonical_coordinates.shape != (n, 3):
            raise ValueError("canonical_coordinates must have shape [n, 3]")
        if self.normalized_coordinates.shape != (n, 3):
            raise ValueError("normalized_coordinates must have shape [n, 3]")


@dataclass(frozen=True)
class BiologicalTissue:
    geometry: IndividualGeometry
    domain_probability: FloatArray
    domain_index: IntArray
    domain_names: NDArray[np.str_]
    cell_type_probability: FloatArray
    cell_type_index: IntArray
    cell_type_names: NDArray[np.str_]
    expression_mean: FloatArray
    expression: FloatArray
    gene_names: NDArray[np.str_]

    def __post_init__(self) -> None:
        n = len(self.geometry.coordinates)
        if len(self.domain_index) != n or len(self.cell_type_index) != n:
            raise ValueError("domain/cell-type outputs must match the number of tissue locations")
        if self.expression.shape[0] != n or self.expression_mean.shape[0] != n:
            raise ValueError("expression outputs must match the number of tissue locations")


@dataclass(frozen=True)
class IdealSection:
    section_id: str
    source_indices: IntArray
    coordinates_2d: FloatArray
    plane_origin: FloatArray
    plane_normal: FloatArray
    expression: FloatArray
    domain_index: IntArray
    cell_type_index: IntArray


@dataclass(frozen=True)
class SimulatedIndividual:
    tissue: BiologicalTissue
    sections: tuple[IdealSection, ...] = ()

    @property
    def individual_id(self) -> str:
        return self.tissue.geometry.individual_id

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tissue = self.tissue
        np.savez_compressed(
            directory / "biological_tissue.npz",
            coordinates=tissue.geometry.coordinates,
            canonical_coordinates=tissue.geometry.canonical_coordinates,
            source_indices=tissue.geometry.truth.source_indices,
            affine=tissue.geometry.truth.affine,
            displacement=tissue.geometry.truth.displacement,
            domain_probability=tissue.domain_probability,
            domain_index=tissue.domain_index,
            domain_names=tissue.domain_names,
            cell_type_probability=tissue.cell_type_probability,
            cell_type_index=tissue.cell_type_index,
            cell_type_names=tissue.cell_type_names,
            expression_mean=tissue.expression_mean,
            expression=tissue.expression,
            gene_names=tissue.gene_names,
        )
        for section in self.sections:
            np.savez_compressed(
                directory / f"{section.section_id}.npz",
                source_indices=section.source_indices,
                coordinates_2d=section.coordinates_2d,
                plane_origin=section.plane_origin,
                plane_normal=section.plane_normal,
                expression=section.expression,
                domain_index=section.domain_index,
                cell_type_index=section.cell_type_index,
            )


@dataclass(frozen=True)
class CaseControlDataset:
    control: tuple[SimulatedIndividual, ...]
    disease: tuple[SimulatedIndividual, ...]

    @property
    def sample_size(self) -> int:
        return len(self.control)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        for group_name, group in (("control", self.control), ("disease", self.disease)):
            for sample_index, individual in enumerate(group):
                individual.save(directory / group_name / f"sample_{sample_index:04d}")
