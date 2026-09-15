"""Approximate Bregma-anchored registration for adata_backup.h5ad.

This is the agreed MVP registration, not a substitute for image-based ANTs
registration.  It creates ``obsm['ccf_coordinates']`` and ``obs['ccf_in_roi']``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from simst import CCFv3Atlas, CCFv3Config
from simst.registration import BregmaRegistrationConfig, register_h5ad_file

HY_ID = 1097


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to adata_backup.h5ad")
    parser.add_argument("--output", default="data/adata_backup_ccf_hy.h5ad")
    parser.add_argument("--annotation", default="data/ccfv3/annotation_10.nrrd")
    parser.add_argument("--tree", default="data/ccfv3/structure_tree.json")
    parser.add_argument("--ml-flip", action="store_true")
    parser.add_argument("--dv-flip", action="store_true")
    args = parser.parse_args()

    atlas = CCFv3Atlas(
        CCFv3Config(
            annotation_path=args.annotation,
            structure_tree_path=args.tree,
            roi_structure_ids=(HY_ID,),
            ccf_voxel_resolution=10,
            domain_granularity="roi_root",
            max_template_voxels=250_000,
        )
    )
    cfg = BregmaRegistrationConfig(ml_flip=args.ml_flip, dv_flip=args.dv_flip)
    output = register_h5ad_file(args.input, args.output, atlas, cfg)
    print(f"Registered AnnData written to: {Path(output)}")
    print("Inspect obs['ccf_in_roi'] and spatial plots before treating this as anatomical truth.")


if __name__ == "__main__":
    main()
