"""Pluggable gene-expression interface and notebook NB2 implementation."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .cell_type import CellTypeOutput
from .data import CanonicalAnatomy, IndividualGeometry, ReferenceData
from .domain import DomainOutput


@dataclass(frozen=True)
class GeneExpressionOutput:
    mean: np.ndarray
    value: np.ndarray
    gene_names: np.ndarray


class GeneExpressionModule(ABC):
    """Cell/domain-to-expression component used by :class:`simst.model.SimST`."""

    @abstractmethod
    def fit(
        self,
        reference: ReferenceData,
        canonical: CanonicalAnatomy,
    ) -> "GeneExpressionModule":
        pass

    @abstractmethod
    def sample(
        self,
        geometry: IndividualGeometry,
        domain: DomainOutput,
        cell_type: CellTypeOutput,
        rng: np.random.Generator,
    ) -> GeneExpressionOutput:
        pass


@dataclass(frozen=True)
class SpatialNBConfig:
    n_spatial_basis: int = 30
    min_nonzero: int = 10
    random_seed: int = 123
    poisson_maxiter: int = 50
    nb_maxiter: int = 100
    library_size_mode: str = "resample_reference"


class SpatialNBGeneExpression(GeneExpressionModule):
    """Learn P(expression | observed cell type, CCF domain, 3D CCF position)."""

    def __init__(self, config: SpatialNBConfig | None = None):
        self.config = config or SpatialNBConfig()
        self.gene_names: np.ndarray | None = None
        self.celltype_levels: np.ndarray | None = None
        self.domain_levels: np.ndarray | None = None
        self.base_columns: list[str] = []
        self.basis_info: dict | None = None
        self.fits: dict[str, dict | None] = {}
        self.fit_status: dict[str, str] = {}
        self.failed_genes: dict[str, str] = {}
        self.reference_library_size: np.ndarray | None = None

    @staticmethod
    def _build_base_matrix(cell_types: np.ndarray, domains: np.ndarray, ct_levels, dom_levels):
        import pandas as pd

        ct = pd.Categorical(cell_types, categories=ct_levels)
        dom = pd.Categorical(domains, categories=dom_levels)
        if np.any(pd.isna(ct)) or np.any(pd.isna(dom)):
            raise ValueError("Expression model received cell-type/domain levels unseen in training")
        ct_df = pd.get_dummies(ct, prefix="celltype", drop_first=True, dtype=float)
        dom_df = pd.get_dummies(dom, prefix="domain", drop_first=True, dtype=float)
        intercept = pd.DataFrame({"Intercept": np.ones(len(cell_types), dtype=float)})
        return pd.concat([intercept, ct_df, dom_df], axis=1)

    def _fit_basis(self, coordinates: np.ndarray):
        from sklearn.cluster import MiniBatchKMeans

        coordinates = np.asarray(coordinates, dtype=np.float64)
        mean = coordinates.mean(axis=0)
        sd = coordinates.std(axis=0)
        standardized = (coordinates - mean) / (sd + 1e-8)
        n_basis = min(int(self.config.n_spatial_basis), len(standardized))
        if n_basis < len(standardized):
            km = MiniBatchKMeans(
                n_clusters=n_basis,
                random_state=self.config.random_seed,
                batch_size=min(4096, len(standardized)),
                n_init="auto",
            )
            centers = km.fit(standardized).cluster_centers_
        else:
            centers = standardized.copy()
        center_dist = np.sqrt(
            np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        )
        positive = center_dist[center_dist > 0]
        bandwidth = float(np.median(positive)) if positive.size else 1.0
        bandwidth = max(bandwidth, 1e-6)
        self.basis_info = {
            "coord_mean": mean,
            "coord_sd": sd,
            "centers": centers,
            "bandwidth": bandwidth,
        }
        return self._basis(coordinates)

    def _basis(self, coordinates: np.ndarray) -> np.ndarray:
        if self.basis_info is None:
            raise RuntimeError("Spatial basis has not been fitted")
        info = self.basis_info
        standardized = (
            np.asarray(coordinates, dtype=float) - info["coord_mean"]
        ) / (info["coord_sd"] + 1e-8)
        d2 = np.sum(
            (standardized[:, None, :] - info["centers"][None, :, :]) ** 2,
            axis=2,
        )
        return np.exp(-0.5 * d2 / float(info["bandwidth"]) ** 2)

    def _fit_one_gene(self, y: np.ndarray, x: np.ndarray, offset: np.ndarray) -> dict:
        import statsmodels.api as sm

        y = np.rint(np.asarray(y, dtype=np.float64)).copy()
        y[y < 0] = 0
        if np.count_nonzero(y) < self.config.min_nonzero or np.var(y) < 1e-12:
            raise ValueError("Gene too sparse or constant for spatial NB-GLM fitting")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poisson_fit = sm.GLM(
                y,
                x,
                family=sm.families.Poisson(),
                offset=offset,
            ).fit(maxiter=self.config.poisson_maxiter, disp=0)
            mu0 = np.maximum(np.asarray(poisson_fit.fittedvalues, dtype=float), 1e-8)
            denominator = float(np.sum(mu0**2))
            alpha_raw = float(np.sum((y - mu0) ** 2 - mu0) / denominator)
            alpha = float(np.clip(alpha_raw, 1e-8, 100.0))
            nb_fit = sm.GLM(
                y,
                x,
                family=sm.families.NegativeBinomial(alpha=alpha),
                offset=offset,
            ).fit(maxiter=self.config.nb_maxiter, disp=0)
        beta = np.asarray(nb_fit.params, dtype=float)
        if not np.isfinite(beta).all():
            raise FloatingPointError("NB2 GLM produced non-finite coefficients")
        return {
            "beta": beta,
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "poisson_converged": bool(getattr(poisson_fit, "converged", True)),
            "nb_converged": bool(getattr(nb_fit, "converged", True)),
        }

    def fit(self, reference: ReferenceData, canonical: CanonicalAnatomy) -> "SpatialNBGeneExpression":
        if reference.library_size is None:
            raise ValueError(
                "SpatialNBGeneExpression requires ReferenceData.library_size. "
                "For the 20260831 data this should be supplied from adata.obs['size']."
            )
        self.gene_names = np.asarray(reference.gene_names, dtype=str)
        self.celltype_levels = np.asarray(sorted(np.unique(reference.cell_type_labels).tolist()), dtype=str)
        self.domain_levels = np.asarray(sorted(np.unique(reference.domain_labels).tolist()), dtype=str)
        self.reference_library_size = np.asarray(reference.library_size, dtype=float)

        base = self._build_base_matrix(
            reference.cell_type_labels,
            reference.domain_labels,
            self.celltype_levels,
            self.domain_levels,
        )
        self.base_columns = list(map(str, base.columns))
        basis = self._fit_basis(reference.ccf_coordinates)
        x = np.concatenate([base.to_numpy(dtype=float), basis], axis=1)
        offset = np.log(np.maximum(self.reference_library_size, 1e-8))

        self.fits = {}
        self.fit_status = {}
        self.failed_genes = {}
        for j, gene in enumerate(self.gene_names):
            try:
                fit = self._fit_one_gene(reference.expression[:, j], x, offset)
                self.fits[str(gene)] = fit
                self.fit_status[str(gene)] = (
                    "success"
                    if fit["poisson_converged"] and fit["nb_converged"]
                    else "nonconverged"
                )
            except Exception as exc:
                gene_name = str(gene)
                self.fits[gene_name] = None
                self.fit_status[gene_name] = "failed"
                self.failed_genes[gene_name] = f"{type(exc).__name__}: {exc}"
        return self

    def _sample_library_size(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.config.library_size_mode != "resample_reference":
            raise ValueError("MVP supports only library_size_mode='resample_reference'")
        if self.reference_library_size is None:
            raise RuntimeError("Reference library sizes are unavailable")
        index = rng.choice(len(self.reference_library_size), size=n, replace=True)
        return self.reference_library_size[index]

    def sample(
        self,
        geometry: IndividualGeometry,
        domain: DomainOutput,
        cell_type: CellTypeOutput,
        rng: np.random.Generator,
    ) -> GeneExpressionOutput:
        if self.gene_names is None or self.celltype_levels is None or self.domain_levels is None:
            raise RuntimeError("SpatialNBGeneExpression must be fitted before sampling")
        from scipy.stats import nbinom

        domain_labels = np.asarray(domain.names, dtype=str)[domain.index]
        cell_labels = np.asarray(cell_type.names, dtype=str)[cell_type.index]
        base = self._build_base_matrix(
            cell_labels,
            domain_labels,
            self.celltype_levels,
            self.domain_levels,
        )
        if list(map(str, base.columns)) != self.base_columns:
            raise RuntimeError("Expression design-matrix columns do not match training columns")
        basis = self._basis(geometry.canonical_coordinates)
        x = np.concatenate([base.to_numpy(dtype=float), basis], axis=1)
        library_size = self._sample_library_size(len(x), rng)
        offset = np.log(np.maximum(library_size, 1e-8))

        mean = np.zeros((len(x), len(self.gene_names)), dtype=float)
        value = np.zeros((len(x), len(self.gene_names)), dtype=np.int32)
        for j, gene in enumerate(self.gene_names):
            fit = self.fits.get(str(gene))
            if fit is None:
                continue
            eta = np.clip(x @ fit["beta"] + offset, -20, 20)
            mu = np.exp(eta)
            alpha = float(fit["alpha"])
            r = 1.0 / max(alpha, 1e-8)
            p = np.clip(r / (r + mu), 1e-12, 1.0 - 1e-12)
            mean[:, j] = mu
            value[:, j] = nbinom(n=r, p=p).rvs(random_state=rng).astype(np.int32)
        return GeneExpressionOutput(mean=mean, value=value, gene_names=self.gene_names)
