# simST

simST is a simulator for multi-individual 3D spatial transcriptomics.

The project learns spatial organization from observed spatial transcriptomics data and generates complete simulated 3D tissues, which can then be virtually sectioned into 2D slices.

The current implementation models three biological levels:

- **Domain**: where anatomical or spatial regions are located;
- **Cell Type**: which cell types are expected at each spatial position;
- **Gene Expression**: how gene expression varies with domain, cell type, and 3D position.

The overall simulation process is:

**3D anatomy → Domain → Cell Type → Gene Expression → 3D tissue → 2D sections**

simST currently supports two execution modes:

| Mode | Canonical 3D space | Domain source | Entry point |
|---|---|---|---|
| **CCFv3** | Allen CCFv3 | Deterministic atlas lookup | `examples/full_ccfv3_hy_mvp.py` |
| **No-CCF** | 3D coordinates in AnnData | `SpatialDomainMLP` | `examples/full_no_ccf_mvp.py` |

---

## Installation

Python 3.11+ is recommended.

```bash
conda create -n stsim python=3.11 -y
conda activate stsim

python -m pip install -U pip
python -m pip install -e '.[full]'
```

---

## Data

Download the project data from Google Drive:

https://drive.google.com/drive/folders/1Dme_UzJi1wTF_aiJrZSEfhtbzvn1SRbm?usp=sharing

Place the downloaded files under `data/`.

```text
data/
├── adata_backup.h5ad
├── hypothalamic_preoptic.h5ad
└── Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv
```

The main example input is:

```text
data/adata_backup.h5ad
```

---

# CCFv3 mode

Use this mode when Allen CCFv3 is available.

The CCFv3 mode uses the Allen Mouse Brain Common Coordinate Framework as the canonical anatomical space. The current MVP uses the full Hypothalamus (HY, Allen structure ID `1097`) as the simulation ROI.

### 1. Download CCFv3

```bash
python scripts/download_ccfv3.py --out data/ccfv3
```

Expected files:

```text
data/ccfv3/
├── annotation_10.nrrd
└── structure_tree.json
```

### 2. Register the training data

```bash
python scripts/register_20260831_to_ccf.py   data/adata_backup.h5ad   --output data/adata_backup_ccf_hy.h5ad
```

### 3. Train and simulate

```bash
python examples/full_ccfv3_hy_mvp.py   --control data/adata_backup.h5ad   --ccf-dir data/ccfv3   --sample-size 3   --work-dir runs/hy_control
```

---

# No-CCF mode

Use this mode when CCFv3 is unavailable but the input AnnData already contains 3D coordinates and domain labels.

The No-CCF mode directly uses the observed 3D coordinates as the canonical spatial support and trains `SpatialDomainMLP` to learn spatial domain distributions.

Default required fields:

```text
obs:    pred_region, cell_type, batch, size
obsm:   spatial_3d_μm_rotate
layers: counts
```

Pipeline:

**Observed 3D coordinates → SpatialDomainMLP → SpatialCellTypeMLP → SpatialNBGeneExpression → 3D tissue → 2D sections**

### Train and simulate

```bash
python examples/full_no_ccf_mvp.py   --control data/adata_backup.h5ad   --work-dir runs/no_ccf_full   --sample-size 3   --domain-epochs 100   --celltype-epochs 300
```

---

## Output

Each simulated individual contains:

```text
biological_tissue.npz
section_000.npz
section_001.npz
...
section_011.npz
```

`biological_tissue.npz` stores the complete simulated 3D tissue, including spatial coordinates, domain assignments, cell-type assignments, and gene-expression values.

The `section_*.npz` files are virtual 2D sections sampled from the simulated 3D tissue.

The fitted model is saved as:

```text
<work-dir>/fitted_model/model.pkl
```

