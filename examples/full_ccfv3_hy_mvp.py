"""End-to-end CCFv3-HY MVP for the 20260831 serial-section data.

Workflow
--------
1. Download CCFv3 once with scripts/download_ccfv3.py.
2. Approximate-register adata_backup.h5ad to CCFv3 with Bregma + rotated 2D
   micrometre coordinates (unless a registered file already exists).
3. Use CCFv3 deterministically for coordinate -> HY domain (no domain training).
4. Train P(cell type | domain, CCF position) from 15-NN 3D targets.
5. Train the spatial NB gene-expression model from the same control data.
6. Generate complete 3D tissues and 12 calibrated 2D sections.
7. If --disease is supplied, preserve the simple DifferenceIntervention path
   and generate matched control/disease pairs.

The registration step is intentionally an MVP approximation.  It is not a
replacement for image-based ANTs/diffeomorphic atlas registration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simst import (
    CCFv3Atlas,
    CCFv3CanonicalBuilder,
    CCFv3Config,
    CCFv3DomainModule,
    CellTypeMLPConfig,
    DifferenceIntervention,
    FittedSimST,
    GeometryConfig,
    SectionConfig,
    SimST,
    SimulationConfig,
    SpatialCellTypeMLP,
    SpatialNBConfig,
    SpatialNBGeneExpression,
)
from simst.io import reference_from_anndata
from simst.registration import BregmaRegistrationConfig, register_h5ad_file

HY_ID = 1097
TRAINING_SECTION_COUNTS = (4832, 4787, 5169, 5070, 5343, 5338, 5488, 5557, 5926, 5803, 5543, 5517)
TOTAL_TRAINING_CELLS = sum(TRAINING_SECTION_COUNTS)  # 64373


def ensure_registered(source: Path, target: Path, atlas: CCFv3Atlas) -> Path:
    if target.exists():
        return target
    print(f"Registering {source} -> {target}")
    return register_h5ad_file(
        source,
        target,
        atlas,
        BregmaRegistrationConfig(
            bregma_key="Bregma",
            section_key="batch",
            spatial_2d_key="spatial_2d_μm_rotate",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, help="adata_backup.h5ad for the baseline group")
    parser.add_argument("--disease", default=None, help="Optional disease/reference h5ad")
    parser.add_argument("--ccf-dir", default="data/ccfv3")
    parser.add_argument("--work-dir", default="runs/ccfv3_hy_mvp")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--reuse-fitted-model",
        action="store_true",
        help="Reuse WORK_DIR/fitted_model/model.pkl if it already exists instead of retraining.",
    )
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ccf_dir = Path(args.ccf_dir)

    # Full Hypothalamus (HY) ROI, 10-um CCF.  The coarsest MVP granularity maps
    # every HY descendant back to the single HY root domain.  Later, change
    # domain_granularity to "native", an ancestor-step integer, or selected IDs.
    atlas = CCFv3Atlas(
        CCFv3Config(
            annotation_path=ccf_dir / "annotation_10.nrrd",
            structure_tree_path=ccf_dir / "structure_tree.json",
            roi_structure_ids=(HY_ID,),
            ccf_voxel_resolution=10,
            domain_granularity="roi_root",
            roi_bounds_um=None,
            # The ROI is still the full HY.  This only subsamples its voxel cloud
            # used as the canonical point-sampling support to keep laptop memory reasonable.
            max_template_voxels=250_000,
            template_seed=args.seed,
        )
    )

    control_path = Path(args.control)
    control_reg = ensure_registered(control_path, work / "control_ccf_hy.h5ad", atlas)
    control_reference = reference_from_anndata(control_reg, filter_to_roi=True)

    engine = SimST(
        canonical_builder=CCFv3CanonicalBuilder(atlas),
        domain_module=CCFv3DomainModule(
            atlas,
            require_full_roi_training_coverage=False,
        ),
        cell_type_module=SpatialCellTypeMLP(
            CellTypeMLPConfig(
                neighborhood_mode="hard",
                hard_knn_k=15,
                epochs=args.epochs,
            )
        ),
        gene_expression_module=SpatialNBGeneExpression(
            SpatialNBConfig(library_size_mode="resample_reference")
        ),
    )

    fitted_model_dir = work / "fitted_model"
    fitted_model_file = fitted_model_dir / "model.pkl"
    if args.reuse_fitted_model and fitted_model_file.exists():
        print(f"Reusing fitted model from {fitted_model_dir}")
        model = FittedSimST.load(fitted_model_dir)
    else:
        print(f"Training baseline model with {len(control_reference.coordinates)} HY-mapped cells...")
        model = engine.fit(control_reference)
        model.save(fitted_model_dir)
    print("3D-neighborhood diagnostics:", model.cell_type_module.neighborhood_diagnostics)

    config = SimulationConfig(
        root_seed=args.seed,
        geometry=GeometryConfig(
            n_points=TOTAL_TRAINING_CELLS,
            ccf_subvoxel_jitter=True,
        ),
        sections=SectionConfig(
            n_sections=12,
            thickness=10.0,
            target_counts=TRAINING_SECTION_COUNTS,
        ),
    )

    if args.disease is None:
        controls = engine.simulate_controls(model, args.sample_size, config)
        for i, individual in enumerate(controls):
            individual.save(work / "control" / f"sample_{i:04d}")
        print(f"Saved {len(controls)} simulated control individuals to {work}")
        return

    disease_path = Path(args.disease)
    disease_reg = ensure_registered(disease_path, work / "disease_ccf_hy.h5ad", atlas)
    disease_reference = reference_from_anndata(disease_reg, filter_to_roi=True)
    intervention = DifferenceIntervention.from_references(control_reference, disease_reference)
    dataset = engine.simulate_case_control(
        model=model,
        sample_size=args.sample_size,
        intervention=intervention,
        config=config,
    )
    dataset.save(work / "case_control")
    print(
        f"Saved {dataset.sample_size} matched control + {dataset.sample_size} disease individuals "
        f"to {work / 'case_control'}"
    )


if __name__ == "__main__":
    main()
