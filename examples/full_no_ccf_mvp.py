# -*- coding: utf-8 -*-

"""
End-to-end simST example without CCFv3.

This entry point is intended for legacy / non-atlas 3D ST data that already
contain:
    - observed 3D coordinates
    - domain labels
    - cell-type labels
    - raw count matrix
    - library sizes

Pipeline
--------
1. Load the AnnData directly, without CCF registration.
2. Build canonical anatomy from the observed 3D coordinates.
3. Fit SpatialDomainMLP:
       P(D | x, y, z)
4. Fit SpatialCellTypeMLP:
       P(C | D, x, y, z)
5. Fit SpatialNBGeneExpression:
       P(E | C, D, x, y, z)
6. Sample complete 3D tissues.
7. Produce virtual 2D sections.
8. Optionally generate matched control / disease simulations.

No CCFv3 files are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simst import (
    CellTypeMLPConfig,
    DifferenceIntervention,
    DomainMLPConfig,
    FittedSimST,
    GeometryConfig,
    SectionConfig,
    SimST,
    SimulationConfig,
    SpatialCellTypeMLP,
    SpatialDomainMLP,
    SpatialNBConfig,
    SpatialNBGeneExpression,
)
from simst.io import reference_from_anndata


def load_reference(
    path: str | Path,
    *,
    domain_key: str,
    cell_type_key: str,
    spatial_key: str,
    counts_layer: str,
    library_size_key: str,
    section_key: str,
):
    """
    Load a non-CCF reference.

    ccf_key=None is the important switch:
    simST then uses the observed 3D coordinates directly as the canonical
    spatial support.
    """
    return reference_from_anndata(
        path,
        ccf_key=None,
        domain_key=domain_key,
        experimental_spatial_key=spatial_key,
        cell_type_key=cell_type_key,
        counts_layer=counts_layer,
        library_size_key=library_size_key,
        section_key=section_key,
        filter_to_roi=False,
    )


def infer_section_counts(section_id: np.ndarray | None) -> tuple[int, tuple[int, ...]] | tuple[None, None]:
    """
    Infer the number of sections and their original cell counts.

    The order follows the first appearance of each section label in the
    reference data rather than lexicographic sorting.
    """
    if section_id is None:
        return None, None

    labels = np.asarray(section_id, dtype=str)

    ordered_names: list[str] = []
    seen: set[str] = set()

    for label in labels:
        label = str(label)
        if label not in seen:
            seen.add(label)
            ordered_names.append(label)

    counts = tuple(int(np.sum(labels == name)) for name in ordered_names)
    return len(ordered_names), counts


def build_engine(args) -> SimST:
    """
    Build the legacy / no-CCF three-model pipeline.

    No canonical_builder is supplied. Therefore SimST.fit() automatically
    calls build_canonical_anatomy(reference) using the observed 3D coordinates.
    """
    domain_module = SpatialDomainMLP(
        DomainMLPConfig(
            epochs=args.domain_epochs,
            batch_size=args.domain_batch_size,
            learning_rate=args.domain_lr,
            validation_fraction=0.2,
            split_seed=args.seed,
            training_seed=args.seed,
            device=args.device,
        )
    )

    # slice_radius is the mode documented in simST as reproducing the original
    # notebook-style cell-type neighbourhood target. For a later 3D-KNN variant,
    # change this to neighborhood_mode="hard", hard_knn_k=15.
    cell_type_module = SpatialCellTypeMLP(
        CellTypeMLPConfig(
            neighborhood_mode="slice_radius",
            slice_neighbors=args.slice_neighbors,
            epochs=args.celltype_epochs,
            batch_size=args.celltype_batch_size,
            learning_rate=args.celltype_lr,
            device=args.device,
        )
    )

    gene_expression_module = SpatialNBGeneExpression(
        SpatialNBConfig(
            n_spatial_basis=args.n_spatial_basis,
            min_nonzero=args.min_nonzero,
            random_seed=args.seed,
            library_size_mode="resample_reference",
        )
    )

    return SimST(
        domain_module=domain_module,
        cell_type_module=cell_type_module,
        gene_expression_module=gene_expression_module,
        # IMPORTANT:
        # canonical_builder is intentionally omitted.
        # SimST will build canonical anatomy directly from reference.coordinates.
    )


def build_simulation_config(reference, args) -> SimulationConfig:
    n_sections, target_counts = infer_section_counts(reference.section_id)

    if args.no_sections:
        section_config = None
    else:
        if n_sections is None:
            n_sections = args.n_sections
            target_counts = None

        section_config = SectionConfig(
            n_sections=n_sections,
            thickness=args.section_thickness,
            target_counts=target_counts if args.match_reference_section_counts else None,
        )

    n_points = args.n_points
    if n_points is None:
        n_points = len(reference.coordinates)

    return SimulationConfig(
        root_seed=args.seed,
        geometry=GeometryConfig(
            n_points=n_points,
            # Non-CCF canonical anatomy is a point cloud rather than a CCF voxel
            # volume, so CCF sub-voxel jitter is not needed.
            ccf_subvoxel_jitter=False,
        ),
        sections=section_config,
    )


def print_fit_diagnostics(model: FittedSimST) -> None:
    domain_acc = getattr(model.domain_module, "validation_accuracy", None)
    if domain_acc is not None:
        print(f"Domain validation accuracy: {domain_acc:.4f}")

    cell_diag = getattr(model.cell_type_module, "neighborhood_diagnostics", None)
    if cell_diag:
        print("Cell-type neighborhood diagnostics:", cell_diag)

    expression_status = getattr(model.gene_expression_module, "fit_status", None)
    if expression_status:
        n_success = sum(status == "success" for status in expression_status.values())
        n_nonconverged = sum(status == "nonconverged" for status in expression_status.values())
        n_failed = sum(status == "failed" for status in expression_status.values())
        print(
            "Gene-expression fit summary:",
            {
                "success": n_success,
                "nonconverged": n_nonconverged,
                "failed": n_failed,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full simST pipeline without CCFv3."
    )

    # ------------------------------------------------------------------
    # Input / output
    # ------------------------------------------------------------------
    parser.add_argument(
        "--control",
        required=True,
        help="Control AnnData .h5ad file.",
    )
    parser.add_argument(
        "--disease",
        default=None,
        help="Optional disease AnnData .h5ad file for case-control simulation.",
    )
    parser.add_argument(
        "--work-dir",
        default="runs/no_ccf_mvp",
        help="Output directory.",
    )

    # ------------------------------------------------------------------
    # AnnData field names
    # ------------------------------------------------------------------
    parser.add_argument("--domain-key", default="pred_region")
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--spatial-key", default="spatial_3d_μm_rotate")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--library-size-key", default="size")
    parser.add_argument("--section-key", default="batch")

    # ------------------------------------------------------------------
    # Domain MLP
    # ------------------------------------------------------------------
    parser.add_argument("--domain-epochs", type=int, default=100)
    parser.add_argument("--domain-batch-size", type=int, default=512)
    parser.add_argument("--domain-lr", type=float, default=1e-3)

    # ------------------------------------------------------------------
    # Cell-type MLP
    # ------------------------------------------------------------------
    parser.add_argument("--celltype-epochs", type=int, default=300)
    parser.add_argument("--celltype-batch-size", type=int, default=1024)
    parser.add_argument("--celltype-lr", type=float, default=1e-3)
    parser.add_argument(
        "--slice-neighbors",
        type=int,
        default=10,
        help="Neighbour count used to determine the per-slice radius.",
    )

    # ------------------------------------------------------------------
    # Spatial NB expression model
    # ------------------------------------------------------------------
    parser.add_argument("--n-spatial-basis", type=int, default=30)
    parser.add_argument("--min-nonzero", type=int, default=10)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="Cells per simulated 3D tissue. Default: number of control reference cells.",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="auto")

    # ------------------------------------------------------------------
    # Virtual sectioning
    # ------------------------------------------------------------------
    parser.add_argument(
        "--n-sections",
        type=int,
        default=12,
        help="Used only when section labels are unavailable.",
    )
    parser.add_argument("--section-thickness", type=float, default=10.0)
    parser.add_argument(
        "--match-reference-section-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the observed number of cells in each original section.",
    )
    parser.add_argument(
        "--no-sections",
        action="store_true",
        help="Generate only complete 3D tissues and skip virtual sectioning.",
    )

    # ------------------------------------------------------------------
    # Fitted-model reuse
    # ------------------------------------------------------------------
    parser.add_argument(
        "--reuse-fitted-model",
        action="store_true",
        help="Reuse WORK_DIR/fitted_model/model.pkl if it exists.",
    )

    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load control reference directly from observed 3D coordinates.
    # ------------------------------------------------------------------
    control_reference = load_reference(
        args.control,
        domain_key=args.domain_key,
        cell_type_key=args.cell_type_key,
        spatial_key=args.spatial_key,
        counts_layer=args.counts_layer,
        library_size_key=args.library_size_key,
        section_key=args.section_key,
    )

    print("Loaded control reference")
    print("  cells:", len(control_reference.coordinates))
    print("  genes:", len(control_reference.gene_names))
    print("  domains:", len(np.unique(control_reference.domain_labels)))
    print("  cell types:", len(np.unique(control_reference.cell_type_labels)))
    print("  coordinate source:", args.spatial_key)
    print("  CCFv3: disabled")

    # ------------------------------------------------------------------
    # 2. Build the no-CCF engine.
    # ------------------------------------------------------------------
    engine = build_engine(args)

    # ------------------------------------------------------------------
    # 3. Fit or reload the three-model pipeline.
    # ------------------------------------------------------------------
    fitted_model_dir = work / "fitted_model"
    fitted_model_file = fitted_model_dir / "model.pkl"

    if args.reuse_fitted_model and fitted_model_file.exists():
        print(f"Reusing fitted model from {fitted_model_dir}")
        model = FittedSimST.load(fitted_model_dir)
    else:
        print("Fitting no-CCF simST model...")
        print("  1/3 SpatialDomainMLP")
        print("  2/3 SpatialCellTypeMLP")
        print("  3/3 SpatialNBGeneExpression")

        model = engine.fit(control_reference)
        model.save(fitted_model_dir)
        print(f"Saved fitted model to {fitted_model_dir}")

    print_fit_diagnostics(model)

    # ------------------------------------------------------------------
    # 4. Simulation configuration.
    # ------------------------------------------------------------------
    simulation_config = build_simulation_config(control_reference, args)

    # ------------------------------------------------------------------
    # 5A. Control-only simulation.
    # ------------------------------------------------------------------
    if args.disease is None:
        controls = engine.simulate_controls(
            model=model,
            sample_size=args.sample_size,
            config=simulation_config,
        )

        for i, individual in enumerate(controls):
            output_dir = work / "control" / f"sample_{i:04d}"
            individual.save(output_dir)

        print(
            f"Saved {len(controls)} simulated control individuals to "
            f"{work / 'control'}"
        )
        return

    # ------------------------------------------------------------------
    # 5B. Optional matched case-control simulation.
    # ------------------------------------------------------------------
    disease_reference = load_reference(
        args.disease,
        domain_key=args.domain_key,
        cell_type_key=args.cell_type_key,
        spatial_key=args.spatial_key,
        counts_layer=args.counts_layer,
        library_size_key=args.library_size_key,
        section_key=args.section_key,
    )

    intervention = DifferenceIntervention.from_references(
        control_reference,
        disease_reference,
    )

    dataset = engine.simulate_case_control(
        model=model,
        sample_size=args.sample_size,
        intervention=intervention,
        config=simulation_config,
    )

    output_dir = work / "case_control"
    dataset.save(output_dir)

    print(
        f"Saved {dataset.sample_size} matched control + "
        f"{dataset.sample_size} disease individuals to {output_dir}"
    )


if __name__ == "__main__":
    main()
