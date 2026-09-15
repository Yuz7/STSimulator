"""Input adapters for training data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import ReferenceData


def reference_from_anndata(
    path: str | Path,
    *,
    ccf_key: str | None = "ccf_coordinates",
    domain_key: str | None = None,
    experimental_spatial_key: str = "spatial_3d_μm_rotate",
    cell_type_key: str = "cell_type",
    counts_layer: str = "counts",
    library_size_key: str = "size",
    section_key: str = "batch",
    filter_to_roi: bool = True,
    roi_flag_key: str = "ccf_in_roi",
) -> ReferenceData:
    """Load registered serial-section ST data for simST training.

    For atlas training, ``adata.obsm[ccf_key]`` must contain CCFv3 coordinates.
    For the original 20260831 models, pass ``ccf_key=None`` and
    ``domain_key="pred_region"`` to use the observed 3D coordinates and domain
    labels directly.
    """
    try:
        import anndata as ad
        from scipy import sparse
    except ImportError as exc:
        raise ImportError("AnnData input requires `anndata` and `scipy`") from exc

    adata = ad.read_h5ad(path)
    if ccf_key is not None and ccf_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{ccf_key!r}] is missing. Register the training ST data to CCFv3 first."
        )
    if ccf_key is None and domain_key is None:
        raise ValueError("domain_key is required when ccf_key=None")
    if domain_key is not None and domain_key not in adata.obs:
        raise KeyError(f"adata.obs[{domain_key!r}] is missing")
    if cell_type_key not in adata.obs:
        raise KeyError(f"adata.obs[{cell_type_key!r}] is missing")
    if counts_layer not in adata.layers:
        raise KeyError(f"adata.layers[{counts_layer!r}] is missing")
    if library_size_key not in adata.obs:
        raise KeyError(f"adata.obs[{library_size_key!r}] is missing")

    if filter_to_roi and roi_flag_key in adata.obs:
        keep = np.asarray(adata.obs[roi_flag_key], dtype=bool)
        if not keep.any():
            raise ValueError("No registered training cells fall inside the selected CCF ROI")
        adata = adata[keep].copy()

    ccf = None if ccf_key is None else np.asarray(adata.obsm[ccf_key], dtype=float)
    if experimental_spatial_key in adata.obsm:
        coordinates = np.asarray(adata.obsm[experimental_spatial_key], dtype=float)
        if coordinates.shape[1] > 3:
            coordinates = coordinates[:, :3]
    elif "spatial_3d" in adata.obsm:
        coordinates = np.asarray(adata.obsm["spatial_3d"], dtype=float)
    else:
        if ccf is None:
            raise KeyError(
                f"adata.obsm[{experimental_spatial_key!r}] and adata.obsm['spatial_3d'] are missing"
            )
        coordinates = ccf.copy()

    counts = adata.layers[counts_layer]
    if sparse.issparse(counts):
        counts = counts.toarray()
    counts = np.rint(np.asarray(counts, dtype=float))
    counts[counts < 0] = 0

    section_id = adata.obs[section_key].astype(str).to_numpy() if section_key in adata.obs else None
    return ReferenceData(
        coordinates=coordinates,
        canonical_coordinates=ccf,
        domain_labels=(
            np.repeat("__CCF_UNASSIGNED__", adata.n_obs)
            if domain_key is None
            else adata.obs[domain_key].astype(str).to_numpy()
        ),
        cell_type_labels=adata.obs[cell_type_key].astype(str).to_numpy(),
        expression=counts,
        gene_names=adata.var_names.astype(str).to_numpy(),
        library_size=adata.obs[library_size_key].to_numpy(dtype=float),
        section_id=section_id,
        metadata={
            "source": str(path),
            "ccf_key": ccf_key,
            "filter_to_roi": bool(filter_to_roi),
            "n_training_cells": int(adata.n_obs),
        },
    )
