"""CCFv3 canonical anatomy and deterministic domain assignment.

The MVP deliberately does not perform ST-to-CCF registration.  Training data
must already contain CCF-registered coordinates, in micrometers and in the same
axis/index convention as the supplied annotation volume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .data import CanonicalAnatomy, IndividualGeometry, ReferenceData
from .domain import DomainModule, DomainOutput


@dataclass(frozen=True)
class CCFv3Config:
    annotation_path: str | Path
    structure_tree_path: str | Path
    roi_structure_ids: tuple[int, ...]
    ccf_voxel_resolution: int = 10
    domain_granularity: str | int | tuple[int, ...] = "roi_root"
    roi_bounds_um: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    max_template_voxels: int | None = None
    template_seed: int = 0


class CCFv3Atlas:
    """Minimal local-file CCFv3 adapter.

    Supported annotation formats are ``.npy`` and ``.nrrd``.  NRRD support
    requires the optional ``pynrrd`` dependency.  The structure tree may be a
    list of Allen-style structure dictionaries or a JSON object containing such
    a list under ``msg``.
    """

    def __init__(self, config: CCFv3Config):
        self.config = config
        self.resolution_um = float(config.ccf_voxel_resolution)
        if self.resolution_um <= 0:
            raise ValueError("ccf_voxel_resolution must be positive")
        self.annotation = self._load_annotation(Path(config.annotation_path))
        if self.annotation.ndim != 3:
            raise ValueError("CCF annotation volume must be three-dimensional")
        self.structures = self._load_structures(Path(config.structure_tree_path))
        self.parent = {}
        for sid, info in self.structures.items():
            parent_id = info.get("parent_structure_id")
            if parent_id is None:
                path = info.get("structure_id_path")
                if isinstance(path, str):
                    path = [int(x) for x in path.strip("/").split("/") if x]
                if isinstance(path, (list, tuple)) and len(path) >= 2:
                    # Allen paths include the structure itself as the final ID.
                    parent_id = path[-2]
            self.parent[sid] = int(parent_id) if parent_id is not None else None
        self._roi_ids = self._expand_descendants(set(map(int, config.roi_structure_ids)))
        if not self._roi_ids:
            raise ValueError("roi_structure_ids must identify at least one CCF structure")

    def __reduce__(self):
        """Serialize only the atlas configuration, not the full 3D annotation array.

        The annotation volume can be very large.  Reconstructing the atlas from
        its local CCF files keeps ``FittedSimST.save()`` compact and avoids
        embedding the entire 10-um CCF volume inside ``model.pkl``.  Loading a
        fitted model therefore requires the CCF files referenced by ``config``
        to remain available at the same paths.
        """
        return (self.__class__, (self.config,))

    @staticmethod
    def _load_annotation(path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.int32)
        if suffix == ".nrrd":
            try:
                import nrrd
            except ImportError as exc:
                raise ImportError("Reading .nrrd CCF files requires `pynrrd`.") from exc
            data, _ = nrrd.read(str(path))
            return np.asarray(data, dtype=np.int32)
        raise ValueError("annotation_path must point to a .npy or .nrrd file")

    @staticmethod
    def _load_structures(path: Path) -> dict[int, dict]:
        """Load either flat AllenSDK JSON or hierarchical StructureGraph JSON.

        The direct Allen API ``structure_graph_download/1.json`` is hierarchical
        (nodes contain ``children``).  Older AllenSDK cache files may instead be
        flat.  Supporting both removes the runtime dependency on AllenSDK.
        """
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and "msg" in payload:
            payload = payload["msg"]

        structures: dict[int, dict] = {}

        def visit(item, parent_id=None):
            if not isinstance(item, dict):
                return
            if "id" in item:
                copied = dict(item)
                copied.pop("children", None)
                if copied.get("parent_structure_id") is None and parent_id is not None:
                    copied["parent_structure_id"] = int(parent_id)
                sid = int(copied["id"])
                structures[sid] = copied
                current_parent = sid
            else:
                current_parent = parent_id
            children = item.get("children", [])
            if isinstance(children, list):
                for child in children:
                    visit(child, current_parent)

        if isinstance(payload, dict):
            # A flat dict keyed by structure ID or one hierarchical root.
            if "id" in payload or "children" in payload:
                visit(payload)
            else:
                for item in payload.values():
                    visit(item)
        elif isinstance(payload, list):
            for item in payload:
                visit(item)

        if not structures:
            raise ValueError("No Allen-style structures were found in structure_tree_path")
        return structures

    def _expand_descendants(self, roots: set[int]) -> set[int]:
        if not roots:
            return set()
        descendants = set(roots)
        changed = True
        while changed:
            changed = False
            for sid, parent in self.parent.items():
                if parent in descendants and sid not in descendants:
                    descendants.add(sid)
                    changed = True
        return descendants

    @property
    def roi_ids(self) -> frozenset[int]:
        return frozenset(self._roi_ids)

    def roi_voxel_indices(self) -> np.ndarray:
        mask = np.isin(self.annotation, np.fromiter(self._roi_ids, dtype=np.int64))
        indices = np.argwhere(mask)
        if self.config.roi_bounds_um is not None and len(indices):
            coordinates = self.indices_to_coordinates(indices)
            bounds = np.asarray(self.config.roi_bounds_um, dtype=float)
            if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
                raise ValueError("roi_bounds_um must contain three (low, high) pairs")
            keep = np.all(
                (coordinates >= bounds[:, 0]) & (coordinates <= bounds[:, 1]),
                axis=1,
            )
            indices = indices[keep]
        if len(indices) == 0:
            raise ValueError("The requested ROI contains no voxels in the annotation volume")
        limit = self.config.max_template_voxels
        if limit is not None and len(indices) > limit:
            rng = np.random.default_rng(self.config.template_seed)
            keep = rng.choice(len(indices), size=int(limit), replace=False)
            indices = indices[keep]
        return indices.astype(np.int64)

    def indices_to_coordinates(self, indices: np.ndarray) -> np.ndarray:
        # Coordinates refer to voxel centers.  Training CCF coordinates must use
        # the same axis convention as this annotation array.
        return (np.asarray(indices, dtype=float) + 0.5) * self.resolution_um

    def coordinates_valid_mask(self, coordinates_um: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates_um, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("CCF coordinates must have shape [n, 3]")
        indices = np.floor(coordinates / self.resolution_um).astype(np.int64)
        return np.all((indices >= 0) & (indices < np.asarray(self.annotation.shape)), axis=1)

    def coordinates_to_indices(self, coordinates_um: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates_um, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("CCF coordinates must have shape [n, 3]")
        indices = np.floor(coordinates / self.resolution_um).astype(np.int64)
        valid = np.all((indices >= 0) & (indices < np.asarray(self.annotation.shape)), axis=1)
        if not valid.all():
            bad = int((~valid).sum())
            raise ValueError(f"{bad} CCF coordinates fall outside the annotation volume")
        return indices

    def lookup_native_ids(self, coordinates_um: np.ndarray) -> np.ndarray:
        idx = self.coordinates_to_indices(coordinates_um)
        return self.annotation[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int64)

    def _ancestor_after_steps(self, sid: int, steps: int) -> int:
        current = int(sid)
        for _ in range(max(0, int(steps))):
            parent = self.parent.get(current)
            if parent is None:
                break
            current = int(parent)
        return current

    def _nearest_selected_ancestor(self, sid: int, selected: set[int]) -> int:
        current = int(sid)
        while True:
            if current in selected:
                return current
            parent = self.parent.get(current)
            if parent is None:
                return int(sid)
            current = int(parent)

    def collapse_ids(self, native_ids: np.ndarray) -> np.ndarray:
        granularity = self.config.domain_granularity
        native_ids = np.asarray(native_ids, dtype=np.int64)
        if granularity == "native":
            return native_ids.copy()
        if granularity == "roi_root":
            roots = set(map(int, self.config.roi_structure_ids))
            return np.asarray(
                [self._nearest_selected_ancestor(int(sid), roots) for sid in native_ids],
                dtype=np.int64,
            )
        if isinstance(granularity, int):
            return np.asarray(
                [self._ancestor_after_steps(int(sid), granularity) for sid in native_ids],
                dtype=np.int64,
            )
        selected = set(map(int, granularity))
        if not selected:
            raise ValueError("domain_granularity structure-ID set cannot be empty")
        return np.asarray(
            [self._nearest_selected_ancestor(int(sid), selected) for sid in native_ids],
            dtype=np.int64,
        )

    def label_for_id(self, sid: int) -> str:
        info = self.structures.get(int(sid), {})
        return str(info.get("acronym") or info.get("name") or int(sid))

    def domain_labels(self, coordinates_um: np.ndarray) -> np.ndarray:
        native = self.lookup_native_ids(coordinates_um)
        collapsed = self.collapse_ids(native)
        return np.asarray([self.label_for_id(int(sid)) for sid in collapsed], dtype=str)

    def roi_domain_labels(self) -> np.ndarray:
        indices = self.roi_voxel_indices()
        native = self.annotation[indices[:, 0], indices[:, 1], indices[:, 2]]
        collapsed = self.collapse_ids(native)
        return np.asarray([self.label_for_id(int(sid)) for sid in collapsed], dtype=str)


class CCFv3CanonicalBuilder:
    """Build the canonical 3D tissue support from a CCFv3 ROI."""

    def __init__(self, atlas: CCFv3Atlas):
        self.atlas = atlas

    def build(self, reference: ReferenceData) -> CanonicalAnatomy:
        indices = self.atlas.roi_voxel_indices()
        coordinates = self.atlas.indices_to_coordinates(indices)
        center = np.median(coordinates, axis=0)
        low, high = np.quantile(coordinates, [0.01, 0.99], axis=0)
        scale = high - low
        if (scale == 0).any():
            raise ValueError("The selected CCF ROI must vary along all three axes")
        normalized = (coordinates - center) / scale
        domains = self.atlas.roi_domain_labels()
        return CanonicalAnatomy(
            coordinates=coordinates,
            normalized_coordinates=normalized,
            center=center,
            scale=scale,
            domain_names=np.unique(domains),
            cell_type_names=np.unique(reference.cell_type_labels),
            gene_names=reference.gene_names.copy(),
            voxel_resolution_um=self.atlas.resolution_um,
            metadata={
                "source": "CCFv3",
                "roi_structure_ids": tuple(self.atlas.config.roi_structure_ids),
                "domain_granularity": self.atlas.config.domain_granularity,
            },
        )


class CCFv3DomainModule(DomainModule):
    """Deterministic CCF coordinate -> anatomical-domain mapping; no training."""

    def __init__(self, atlas: CCFv3Atlas, require_full_roi_training_coverage: bool = False):
        self.atlas = atlas
        self.require_full_roi_training_coverage = bool(require_full_roi_training_coverage)
        self.names: np.ndarray | None = None

    def fit(self, reference: ReferenceData, canonical: CanonicalAnatomy) -> "CCFv3DomainModule":
        self.names = np.asarray(canonical.domain_names, dtype=str)
        return self

    def annotate_reference(
        self,
        reference: ReferenceData,
        canonical: CanonicalAnatomy,
    ) -> ReferenceData:
        labels = self.atlas.domain_labels(reference.ccf_coordinates)
        training_domains = set(labels.tolist())
        roi_domains = set(np.asarray(canonical.domain_names, dtype=str).tolist())
        if self.require_full_roi_training_coverage:
            missing = sorted(roi_domains - training_domains)
            if missing:
                preview = ", ".join(missing[:20])
                suffix = "..." if len(missing) > 20 else ""
                raise ValueError(
                    "The CCF simulation ROI contains domains that are absent from the training "
                    f"ST data: {preview}{suffix}.  For the MVP, use a training-matched ROI or "
                    "set require_full_roi_training_coverage=False only if you intentionally "
                    "accept extrapolation."
                )
        return replace(reference, domain_labels=labels)

    def sample(
        self,
        geometry: IndividualGeometry,
        rng: np.random.Generator,
    ) -> DomainOutput:
        if self.names is None:
            raise RuntimeError("CCFv3DomainModule must be fitted before sampling")
        labels = self.atlas.domain_labels(geometry.canonical_coordinates)
        lookup = {name: index for index, name in enumerate(self.names.tolist())}
        unknown = sorted(set(labels.tolist()) - set(lookup))
        if unknown:
            raise ValueError(f"Generated canonical coordinates contain unknown ROI domains: {unknown}")
        index = np.asarray([lookup[name] for name in labels], dtype=np.int64)
        probability = np.eye(len(self.names), dtype=float)[index]
        return DomainOutput(probability=probability, index=index, names=self.names)
