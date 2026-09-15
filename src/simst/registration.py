"""Approximate Bregma-anchored registration of serial coronal ST sections to CCFv3.

This module is intentionally an MVP registration layer.  It is designed to make
serial 2D sections usable by the CCFv3-informed simulator without pretending to
replace image-based ANTs/diffeomorphic registration.

Assumptions for the 20260831 MERFISH data:
- ``adata.obs['Bregma']`` is in millimetres relative to Bregma.
- ``adata.obsm['spatial_2d_μm_rotate']`` contains upright in-plane coordinates
  in micrometres.
- Sections are approximately coronal.

The commonly used estimated Bregma location in the 10-um Allen CCFv3 grid is
(AP, DV, ML) = (540, 44, 570) voxels.  The estimate is approximate and should
not be interpreted as publication-grade stereotaxic registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ccfv3 import CCFv3Atlas


@dataclass(frozen=True)
class BregmaRegistrationConfig:
    bregma_key: str = "Bregma"
    section_key: str = "batch"
    spatial_2d_key: str = "spatial_2d_μm_rotate"
    output_key: str = "ccf_coordinates"
    # Estimated CCFv3 Bregma in AP/DV/ML voxel order at 10 um.
    bregma_ccf_voxel_10um: tuple[float, float, float] = (540.0, 44.0, 570.0)
    # Positive stereotaxic Bregma values are anterior to Bregma, whereas CCF AP
    # indices increase posteriorly, hence the default minus sign.
    ap_sign: float = -1.0
    ml_flip: bool = False
    dv_flip: bool = False
    inplane_scale: float = 1.0
    # Keep the already-calibrated micrometre scale by default.  Translation is
    # estimated independently for each section by centering the section on the
    # HY cross-section (ML midline + median HY DV).
    center_ml_on_ccf_midline: bool = True
    center_dv_on_roi_median: bool = True


def _section_bregma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("A section has no finite Bregma values")
    return float(np.median(values))


def _roi_dv_median_at_ap(atlas: CCFv3Atlas, ap_um: float) -> float:
    """Median DV coordinate of the selected ROI at the nearest AP plane."""
    ap_index = int(np.floor(ap_um / atlas.resolution_um))
    ap_index = int(np.clip(ap_index, 0, atlas.annotation.shape[0] - 1))
    plane = atlas.annotation[ap_index]
    roi_ids = np.fromiter(atlas.roi_ids, dtype=np.int64)
    dv_ml = np.argwhere(np.isin(plane, roi_ids))
    if len(dv_ml) == 0:
        # Search nearby planes because a Bregma estimate can land just outside
        # a thin edge of the ROI.
        for radius in range(1, 51):
            for candidate in (ap_index - radius, ap_index + radius):
                if 0 <= candidate < atlas.annotation.shape[0]:
                    plane = atlas.annotation[candidate]
                    dv_ml = np.argwhere(np.isin(plane, roi_ids))
                    if len(dv_ml):
                        return float(np.median((dv_ml[:, 0] + 0.5) * atlas.resolution_um))
        raise ValueError(
            "No selected-ROI voxels were found near the AP coordinate implied by Bregma"
        )
    return float(np.median((dv_ml[:, 0] + 0.5) * atlas.resolution_um))


def register_serial_sections_to_ccfv3(
    adata,
    atlas: CCFv3Atlas,
    config: BregmaRegistrationConfig | None = None,
):
    """Return a copy of AnnData with approximate CCF coordinates.

    The returned coordinate order is ``(AP, DV, ML)`` in micrometres, matching
    the annotation-array order used by :class:`CCFv3Atlas`.
    """
    cfg = config or BregmaRegistrationConfig()
    if cfg.bregma_key not in adata.obs:
        raise KeyError(f"adata.obs[{cfg.bregma_key!r}] is missing")
    if cfg.section_key not in adata.obs:
        raise KeyError(f"adata.obs[{cfg.section_key!r}] is missing")
    if cfg.spatial_2d_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{cfg.spatial_2d_key!r}] is missing")

    xy = np.asarray(adata.obsm[cfg.spatial_2d_key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) != adata.n_obs:
        raise ValueError(f"adata.obsm[{cfg.spatial_2d_key!r}] must have shape [n_obs, >=2]")

    section = adata.obs[cfg.section_key].astype(str).to_numpy()
    bregma = np.asarray(adata.obs[cfg.bregma_key], dtype=float)
    out = np.zeros((adata.n_obs, 3), dtype=float)

    # Convert the 10-um estimated Bregma anchor to the atlas resolution.
    bregma_ap_um = cfg.bregma_ccf_voxel_10um[0] * 10.0
    bregma_ml_um = cfg.bregma_ccf_voxel_10um[2] * 10.0

    for sid in np.unique(section):
        mask = section == sid
        bg_mm = _section_bregma(bregma[mask])
        ap_um = bregma_ap_um + cfg.ap_sign * bg_mm * 1000.0

        section_xy = xy[mask, :2] * float(cfg.inplane_scale)
        x = section_xy[:, 0]
        y = section_xy[:, 1]
        x0 = float(np.median(x))
        y0 = float(np.median(y))
        ml = x - x0
        dv = y - y0
        if cfg.ml_flip:
            ml = -ml
        if cfg.dv_flip:
            dv = -dv
        if cfg.center_ml_on_ccf_midline:
            ml = ml + bregma_ml_um
        if cfg.center_dv_on_roi_median:
            dv = dv + _roi_dv_median_at_ap(atlas, ap_um)

        out[mask, 0] = ap_um
        out[mask, 1] = dv
        out[mask, 2] = ml

    result = adata.copy()
    result.obsm[cfg.output_key] = out
    valid = atlas.coordinates_valid_mask(out)
    native = np.zeros(len(out), dtype=np.int64)
    if valid.any():
        native[valid] = atlas.lookup_native_ids(out[valid])
    collapsed = atlas.collapse_ids(native)
    result.obs["ccf_structure_id_native"] = native.astype(str)
    result.obs["ccf_domain"] = np.asarray(
        [atlas.label_for_id(int(x)) if valid[i] else "OUTSIDE_CCF" for i, x in enumerate(collapsed)],
        dtype=str,
    )
    result.obs["ccf_in_roi"] = valid & np.isin(
        native, np.fromiter(atlas.roi_ids, dtype=np.int64)
    )
    return result


def register_h5ad_file(
    input_path: str | Path,
    output_path: str | Path,
    atlas: CCFv3Atlas,
    config: BregmaRegistrationConfig | None = None,
) -> Path:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("register_h5ad_file requires `anndata`") from exc
    adata = ad.read_h5ad(input_path)
    registered = register_serial_sections_to_ccfv3(adata, atlas, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    registered.write_h5ad(output)
    return output
