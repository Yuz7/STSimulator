
## v0.3.2 hotfix (2026-09-08)

- Fixed fitted-model serialization by moving `SpatialProbMLP` from a local helper scope to module scope.
- CCFv3 atlas objects now pickle by configuration only, so the full 10-um annotation volume is not embedded in `model.pkl`; it is reloaded from the configured local CCF files on `FittedSimST.load()`.
- Model saving is atomic (`model.pkl.tmp` -> `model.pkl`) to avoid leaving a corrupt partial model after a serialization failure.
- Added a regression test for pickle-safe `SpatialCellTypeMLP`.

# Integration notes: final CCFv3-HY MVP

## Baseline training

1. Approximate-register serial 2D ST cells to CCFv3 coordinates `u=(AP,DV,ML)`.
2. Restrict the reference used for training to cells mapped into the full HY ROI.
3. Assign `D=HY` deterministically using CCFv3 (`domain_granularity='roi_root'`).
4. Build 3D 15-nearest-neighbour local cell-type composition targets, excluding the cell itself.
5. Train the existing MLP formulation for `P(C | D,u)`.
6. Fit the existing spatial NB model for `P(E | C,D,u)` using observed reference cell-type labels and reference library sizes.

## Simulation

1. Sample canonical positions throughout the full HY CCF support.
2. Add sub-voxel jitter for continuous coordinates.
3. Apply individual geometry variation.
4. Assign `D` by CCF lookup.
5. Sample cell types from the fitted MLP.
6. Sample counts from the fitted spatial NB model.
7. Save the complete 3D tissue.
8. Produce 12 2D section outputs.  The MVP uses target-count calibration to match the original section sizes.

## Intervention

Unchanged.  The baseline model is fitted from control only.  The simple `DifferenceIntervention.from_references(control,disease)` remains a separate path and applies global gene-wise disease-minus-control mean differences.

## Fixed defaults

- HY ID: 1097
- CCF voxel resolution: 10 µm
- domain granularity: `roi_root`
- hard 3D neighbourhood: 15-NN
- 3D tissue cells: 64,373
- 12 section target counts: 4832, 4787, 5169, 5070, 5343, 5338, 5488, 5557, 5926, 5803, 5543, 5517

## Dependency note (v0.3.1)

AllenSDK has been removed from the simST environment. AllenSDK 2.16.x pins `numpy<1.24`, which conflicts with simST's modern scientific-Python dependencies. CCFv3 files are now downloaded directly from the official Allen Institute download/API servers using `scripts/download_ccfv3.py`; AllenSDK is not needed for training or simulation.

## v0.3.3 ROI-boundary fix

`IndividualGeometry.canonical_coordinates` now remain in the sampled CCF ROI.
Individual-specific `coordinate_jitter`, affine scaling/rotation, and smooth warp are applied only to the physical tissue coordinates.  This preserves the intended semantics

`CCF canonical coordinate -> anatomical identity -> individual deformation`

and prevents HY samples near the boundary from being relabeled as neighbouring CCF structures.
The end-to-end example also supports `--reuse-fitted-model` so a successfully saved baseline model can be reused after a simulation-stage failure.
