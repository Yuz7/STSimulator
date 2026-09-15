"""Pluggable cell-type interface and notebook MLP implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - torch is a core dependency
    raise ImportError("SpatialCellTypeMLP requires PyTorch") from exc

from .data import CanonicalAnatomy, IndividualGeometry, ReferenceData
from .domain import DomainOutput


@dataclass(frozen=True)
class CellTypeOutput:
    probability: np.ndarray
    index: np.ndarray
    names: np.ndarray


class CellTypeModule(ABC):
    """Domain-to-cell-type component used by :class:`simst.model.SimST`."""

    @abstractmethod
    def fit(self, reference: ReferenceData, canonical: CanonicalAnatomy) -> "CellTypeModule":
        pass

    @abstractmethod
    def sample(
        self,
        geometry: IndividualGeometry,
        domain: DomainOutput,
        rng: np.random.Generator,
    ) -> CellTypeOutput:
        pass


@dataclass(frozen=True)
class CellTypeMLPConfig:
    neighborhood_mode: str = "hard"
    slice_neighbors: int = 10
    hard_knn_k: int = 15
    kernel_bandwidth_um: float | None = None
    kernel_truncate_sigma: float = 3.0
    epochs: int = 300
    batch_size: int = 1024
    learning_rate: float = 1e-3
    device: str = "auto"


class SpatialProbMLP(nn.Module):
    """Pickle-safe PyTorch network for spatial cell-type probabilities.

    This class must live at module scope.  Defining it inside a helper function
    makes fitted simST models impossible to serialize with Python ``pickle``.
    """

    def __init__(self, input_dim: int, n_celltypes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, n_celltypes),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class SpatialCellTypeMLP(CellTypeModule):
    """Learn P(cell type | CCF domain, 3D CCF position).

    ``slice_radius`` reproduces the notebook target: within each observed slice,
    use the mean tenth-neighbour distance as a 2D radius. ``hard`` and ``kernel``
    retain the later 3D CCF alternatives.
    """

    def __init__(self, config: CellTypeMLPConfig | None = None):
        self.config = config or CellTypeMLPConfig()
        self.names: np.ndarray | None = None
        self.domain_names: np.ndarray | None = None
        self.transforms: dict[str, dict] = {}
        self.model = None
        self.input_dim: int | None = None
        self.neighborhood_diagnostics: dict = {}

    def _local_probabilities(
        self, coordinates: np.ndarray, labels: np.ndarray, section_id: np.ndarray | None = None
    ) -> np.ndarray:
        from sklearn.neighbors import KDTree

        mode = self.config.neighborhood_mode.lower()
        if mode not in {"slice_radius", "hard", "kernel"}:
            raise ValueError("neighborhood_mode must be 'slice_radius', 'hard', or 'kernel'")
        names = np.asarray(self.names, dtype=str)
        lookup = {name: i for i, name in enumerate(names.tolist())}
        prob = np.zeros((len(coordinates), len(names)), dtype=float)

        if mode == "slice_radius":
            from sklearn.neighbors import NearestNeighbors

            if section_id is None:
                raise ValueError("slice_radius neighborhoods require ReferenceData.section_id")
            section = np.asarray(section_id, dtype=str)
            n_neighbors = int(self.config.slice_neighbors)
            radii = {}
            for section_name in np.unique(section):
                row = np.flatnonzero(section == section_name)
                xy = np.asarray(coordinates[row, :2], dtype=float)
                if len(xy) < n_neighbors:
                    raise ValueError(
                        f"section {section_name!r} has fewer than {n_neighbors} cells"
                    )
                nearest = NearestNeighbors(n_neighbors=n_neighbors).fit(xy)
                distances, _ = nearest.kneighbors(xy)
                radius = float(distances[:, -1].mean())
                radii[str(section_name)] = radius
                tree = KDTree(xy)
                neighborhoods = tree.query_radius(xy, r=radius)
                for local_i, neighbours in enumerate(neighborhoods):
                    labels_in_neighborhood = labels[row[neighbours]]
                    for label in labels_in_neighborhood:
                        prob[row[local_i], lookup[str(label)]] += 1.0
                    prob[row[local_i]] /= prob[row[local_i]].sum()
            self.neighborhood_diagnostics = {
                "mode": "slice_radius",
                "n_neighbors_for_radius": n_neighbors,
                "radius_by_section": radii,
            }
            return prob

        tree = KDTree(coordinates)

        if mode == "hard":
            k = int(self.config.hard_knn_k)
            if k < 1:
                raise ValueError("hard_knn_k must be >= 1")
            if len(coordinates) <= k:
                raise ValueError("hard_knn_k must be smaller than the number of training cells")
            # k+1 because the first nearest neighbour is the cell itself.
            distance, indices = tree.query(coordinates, k=k + 1)
            indices = indices[:, 1:]
            distance = distance[:, 1:]
            for i, neighbours in enumerate(indices):
                for j in neighbours:
                    prob[i, lookup[str(labels[j])]] += 1.0
                prob[i] /= float(k)
            if section_id is not None:
                sec = np.asarray(section_id, dtype=str)
                cross = np.mean(sec[indices] != sec[:, None])
                self.neighborhood_diagnostics = {
                    "mode": "hard",
                    "k": k,
                    "cross_section_edge_fraction": float(cross),
                    "median_neighbor_distance_um": float(np.median(distance)),
                }
            else:
                self.neighborhood_diagnostics = {"mode": "hard", "k": k}
            return prob

        bandwidth = self.config.kernel_bandwidth_um
        if bandwidth is None or bandwidth <= 0:
            raise ValueError(
                "kernel_bandwidth_um must be set to a positive value for Gaussian-kernel neighborhoods"
            )
        cutoff = float(self.config.kernel_truncate_sigma) * float(bandwidth)
        neighborhoods, distances = tree.query_radius(
            coordinates, r=cutoff, return_distance=True, sort_results=False
        )
        cross_edges = 0
        total_edges = 0
        sec = np.asarray(section_id, dtype=str) if section_id is not None else None
        for i, (indices, distance) in enumerate(zip(neighborhoods, distances)):
            # Exclude the point itself so that the target does not leak its own label.
            keep = indices != i
            indices = indices[keep]
            distance = distance[keep]
            if len(indices) == 0:
                # Fall back to the nearest non-self neighbour.
                d, nn = tree.query(coordinates[i : i + 1], k=min(2, len(coordinates)))
                indices = nn[0, 1:]
                distance = d[0, 1:]
            weights = np.exp(-0.5 * (distance / float(bandwidth)) ** 2)
            for j, weight in zip(indices, weights):
                prob[i, lookup[str(labels[j])]] += float(weight)
            total = prob[i].sum()
            if total <= 0:
                raise RuntimeError("Gaussian-kernel cell-type target has zero total weight")
            prob[i] /= total
            if sec is not None:
                cross_edges += int(np.sum(sec[indices] != sec[i]))
                total_edges += len(indices)
        self.neighborhood_diagnostics = {
            "mode": "kernel",
            "bandwidth_um": float(bandwidth),
            "cross_section_edge_fraction": (
                float(cross_edges / total_edges) if total_edges else float("nan")
            ),
        }
        return prob

    @staticmethod
    def _fit_domain_transforms(coordinates: np.ndarray, domains: np.ndarray) -> tuple[np.ndarray, dict]:
        from sklearn.decomposition import PCA

        transformed = np.zeros_like(coordinates, dtype=float)
        transforms: dict[str, dict] = {}
        for domain in np.unique(domains):
            mask = domains == domain
            xyz = coordinates[mask]
            if len(xyz) < 4:
                pca = None
                projection = xyz.copy()
            else:
                pca = PCA(n_components=3).fit(xyz)
                projection = pca.transform(xyz)
            minimum = projection.min(axis=0)
            scale = projection.max(axis=0) - minimum
            scale[scale < 1e-6] = 1.0
            transformed[mask] = 2.0 * (projection - minimum) / scale - 1.0
            transforms[str(domain)] = {
                "pca": pca,
                "min_xyz": minimum,
                "scale": scale,
            }
        return transformed, transforms

    def _transform_new(self, coordinates: np.ndarray, domains: np.ndarray) -> np.ndarray:
        output = np.zeros_like(coordinates, dtype=float)
        for domain in np.unique(domains):
            key = str(domain)
            if key not in self.transforms:
                raise ValueError(f"No trained cell-type coordinate transform for domain {key!r}")
            mask = domains == domain
            xyz = coordinates[mask]
            transform = self.transforms[key]
            pca = transform["pca"]
            projection = xyz.copy() if pca is None else pca.transform(xyz)
            output[mask] = 2.0 * (projection - transform["min_xyz"]) / transform["scale"] - 1.0
        return output

    def _build_features(self, coordinates: np.ndarray, domains: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            relative, self.transforms = self._fit_domain_transforms(coordinates, domains)
        else:
            relative = self._transform_new(coordinates, domains)
        domain_names = np.asarray(self.domain_names, dtype=str)
        domain_lookup = {name: i for i, name in enumerate(domain_names.tolist())}
        unknown = sorted(set(map(str, domains.tolist())) - set(domain_lookup))
        if unknown:
            raise ValueError(f"Unseen domains in cell-type model: {unknown}")
        domain_index = np.asarray([domain_lookup[str(x)] for x in domains], dtype=np.int64)
        one_hot = np.eye(len(domain_names), dtype=float)[domain_index]
        return np.concatenate([relative, one_hot], axis=1)

    def fit(self, reference: ReferenceData, canonical: CanonicalAnatomy) -> "SpatialCellTypeMLP":
        from torch.utils.data import DataLoader, TensorDataset

        coordinates = np.asarray(reference.ccf_coordinates, dtype=float)
        domains = np.asarray(reference.domain_labels, dtype=str)
        cell_types = np.asarray(reference.cell_type_labels, dtype=str)
        self.names = np.asarray(sorted(np.unique(cell_types).tolist()), dtype=str)
        self.domain_names = np.asarray(sorted(np.unique(domains).tolist()), dtype=str)

        target = self._local_probabilities(
            np.asarray(reference.coordinates, dtype=float),
            cell_types,
            reference.section_id,
        )
        features = self._build_features(coordinates, domains, fit=True)
        self.input_dim = int(features.shape[1])

        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        model = SpatialProbMLP(self.input_dim, len(self.names)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        for _ in range(int(self.config.epochs)):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model = model.to("cpu")
        model.eval()
        self.model = model
        return self

    def sample(
        self,
        geometry: IndividualGeometry,
        domain: DomainOutput,
        rng: np.random.Generator,
    ) -> CellTypeOutput:
        if self.model is None or self.names is None or self.domain_names is None:
            raise RuntimeError("SpatialCellTypeMLP must be fitted before sampling")
        domain_labels = np.asarray(domain.names, dtype=str)[domain.index]
        features = self._build_features(
            np.asarray(geometry.canonical_coordinates, dtype=float),
            domain_labels,
            fit=False,
        )
        with torch.no_grad():
            probability = self.model(torch.tensor(features, dtype=torch.float32)).cpu().numpy()
        probability = probability / probability.sum(axis=1, keepdims=True)
        index = np.asarray(
            [rng.choice(len(self.names), p=probability[i]) for i in range(len(probability))],
            dtype=np.int64,
        )
        return CellTypeOutput(probability=probability, index=index, names=self.names)
