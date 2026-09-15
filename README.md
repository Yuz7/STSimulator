
## v0.3.2 serialization hotfix

The cell-type PyTorch network is now defined at module scope, so a fitted
`FittedSimST` can be saved with `pickle`.  This fixes the error
`Can't pickle local object 'SpatialCellTypeMLP._torch_components.<locals>.SpatialProbMLP'`
that occurred immediately after training in v0.3.1.  The CCF atlas is also
serialized by configuration only (the large annotation volume is reloaded from
the local CCF files on model load), and model saving is atomic so a failed save
does not leave a corrupt `model.pkl`.

# simST CCFv3 MVP

This package integrates the original `simST` framework with the three-model prototype from `20260831 汇总.zip` into the agreed first complete MVP:

\[
\text{CCFv3 anatomy} \rightarrow D \rightarrow C \rightarrow E \rightarrow \text{3D tissue} \rightarrow \text{2D sections}.
\]

The goal of this version is **to run the full training + simulation pipeline reliably before expanding the intervention model**.

## Fixed MVP decisions

- CCFv3 voxel resolution: **10 µm**.
- Simulation ROI: the **full Hypothalamus (HY)** CCF structure, Allen structure ID **1097**.
- Domain granularity: **`roi_root`**, so every HY descendant is collapsed to the single coarse domain `HY`. Change this later when you want finer anatomical domains.
- Coordinate -> domain: **deterministic CCF lookup; no training**.
- Cell-type model: learned from the control ST training data with a **3D 15-nearest-neighbour hard neighborhood** in CCF coordinates.
- Gene-expression model: learned from the control ST data using the existing all-slice spatial negative-binomial formulation.
- Library sizes: resampled from `adata.obs['size']` in the control reference.
- Intervention: unchanged simple `DifferenceIntervention`, learned outside `SimST.fit()` from the global disease-minus-control gene means.
- Output scale: 64,373 cells in each latent 3D tissue and 12 simulated 2D sections with target counts matching the 12 original sections.

## Important scientific limitation of this MVP

The current reference consists of 12 serial 2D MERFISH sections.  The package performs an **approximate Bregma-anchored 2.5D CCF registration** using `Bregma` and `obsm['spatial_2d_μm_rotate']`.  This is sufficient for the MVP software pipeline, but it is **not publication-grade anatomical registration**.  A later version should replace this with image/cell-landmark-based affine + deformable registration.

The full HY is allowed as the simulation ROI even though the 12 training sections cover only part of HY.  Consequently, cell-type and expression generation outside the observed training support is model extrapolation.

## What is Bregma?

Bregma is a skull landmark: the intersection of the coronal and sagittal sutures.  In mouse stereotaxic experiments it is commonly used as the zero reference for the anterior-posterior coordinate.  A section at `Bregma = +0.20 mm` is approximately 0.20 mm anterior to Bregma, while `Bregma = -0.20 mm` is approximately 0.20 mm posterior.  Bregma is a **stereotaxic skull reference**, not an intrinsic CCF voxel label.  The Bregma-to-CCF correspondence used by this MVP is therefore an approximate initialization.

## Installation

Python 3.10+ is supported. AllenSDK is intentionally **not** installed by simST because AllenSDK 2.16.x requires `numpy<1.24`, while simST uses a modern NumPy stack. The CCF files are downloaded directly from the Allen Institute instead.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e '.[full]'
```

## Step 1. Download CCFv3

```bash
python scripts/download_ccfv3.py --out data/ccfv3
```

This downloads the official CCFv3 2017 10-µm annotation and mouse-brain ontology directly from the Allen Institute and creates:

```text
data/ccfv3/annotation_10.nrrd
data/ccfv3/structure_tree.json
```

## Step 2. Register `adata_backup.h5ad` to the HY CCF space

Your training AnnData is expected to contain the fields used by `20260831 汇总.zip`, in particular:

```text
obs: Bregma, cell_type, batch, size, ...
obsm: spatial_2d_μm_rotate, spatial_3d_μm_rotate, ...
layers: counts
```

Run:

```bash
python scripts/register_20260831_to_ccf.py /path/to/adata_backup.h5ad \
  --output data/adata_backup_ccf_hy.h5ad
```

The output contains:

```text
obsm['ccf_coordinates']       # AP, DV, ML in µm
obs['ccf_structure_id_native']
obs['ccf_domain']
obs['ccf_in_roi']             # True when the approximate mapping lands in HY
```

Before trusting the registration scientifically, inspect the fraction of `ccf_in_roi` cells and spatial plots.  If left-right or dorsal-ventral orientation is reversed, rerun with `--ml-flip` and/or `--dv-flip`.

## Step 3. Train + simulate the control model

```bash
python examples/full_ccfv3_hy_mvp.py \
  --control /path/to/adata_backup.h5ad \
  --sample-size 3 \
  --work-dir runs/hy_control
```

The example automatically creates a registered copy if one does not already exist, trains:

\[
P(C\mid D,\mathbf u)
\]

and

\[
P(E\mid C,D,\mathbf u),
\]

then saves `sample-size` independent simulated control individuals.

For each individual:

```text
biological_tissue.npz      # complete 3D simulated tissue
section_000.npz
...
section_011.npz            # 12 simulated 2D sections
```

The cell-type training step prints a diagnostic including the fraction of 15-NN edges that connect cells from different original sections.

## Step 4. Optional simple case-control simulation

The intervention model is intentionally unchanged in this MVP.  Supply a disease reference from the same anatomical context:

```bash
python examples/full_ccfv3_hy_mvp.py \
  --control /path/to/control.h5ad \
  --disease /path/to/disease.h5ad \
  --sample-size 3 \
  --work-dir runs/hy_case_control
```

The baseline model is fitted only from control.  `DifferenceIntervention` separately computes, for each gene,

\[
\Delta_g = \bar Y_g^{Disease} - \bar Y_g^{Control},
\]

and applies this simple global difference to the generated controls.  The result contains matched control/disease pairs.

## Main source files

```text
src/simst/ccfv3.py                 CCF atlas, HY canonical anatomy, domain lookup
src/simst/registration.py          MVP Bregma-anchored serial-section registration
src/simst/modules/cell_type_mlp.py 3D 15-NN / optional Gaussian-kernel cell-type model
src/simst/modules/spatial_nb.py    spatial negative-binomial expression model
src/simst/geometry.py              continuous 3D individual geometry
src/simst/section.py               3D -> 2D section sampling
src/simst/intervention.py          unchanged simple differential intervention
src/simst/model.py                 fit/simulate orchestration
```

## Changing the domain granularity later

The default is the coarsest possible HY-specific choice:

```python
domain_granularity="roi_root"
```

This makes `HY` the only domain.  Later you can use:

- `"native"`: native/fine CCF structure labels;
- an integer: move each native structure upward by that many parent steps;
- a tuple of selected CCF structure IDs: collapse each voxel to its nearest selected ancestor.

This anatomical granularity is different from `ccf_voxel_resolution=10`, which controls the spatial voxel size.

## Hard vs Gaussian-kernel 3D cell-type neighborhoods

Current MVP:

```python
CellTypeMLPConfig(
    neighborhood_mode="hard",
    hard_knn_k=15,
)
```

Future/optional Gaussian kernel:

```python
CellTypeMLPConfig(
    neighborhood_mode="kernel",
    kernel_bandwidth_um=50.0,
)
```

## Tests

The package includes lightweight synthetic tests that do not require CCF or AnnData files:

```bash
pytest -q
```

These are software smoke tests only; they do not validate the biological quality of the CCF registration or generated tissues.

## v0.3.3 ROI-boundary fix

`IndividualGeometry.canonical_coordinates` now remain in the sampled CCF ROI.
Individual-specific `coordinate_jitter`, affine scaling/rotation, and smooth warp are applied only to the physical tissue coordinates.  This preserves the intended semantics

`CCF canonical coordinate -> anatomical identity -> individual deformation`

and prevents HY samples near the boundary from being relabeled as neighbouring CCF structures.
The end-to-end example also supports `--reuse-fitted-model` so a successfully saved baseline model can be reused after a simulation-stage failure.
