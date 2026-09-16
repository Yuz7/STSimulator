# simST CCFv3 MVP

simST is a CCFv3-informed simulator for multi-individual 3D spatial transcriptomics.

This version integrates the original `simST` framework with the three-model prototype from `20260831 汇总.zip` and implements the following pipeline:

**CCFv3 anatomy → Domain → Cell Type → Gene Expression → 3D tissue → 2D sections**

The current MVP uses:

- Allen CCFv3 10-µm atlas;
- the full Hypothalamus (HY, Allen structure ID `1097`) as the simulation ROI;
- deterministic CCF lookup for anatomical domains;
- a 3D spatial cell-type model;
- a spatial negative-binomial gene-expression model;
- 3D tissue generation followed by virtual 2D sectioning.

## 1. Create the environment

Python 3.11+ is recommended.

```bash
conda create -n stsim python=3.11 -y
conda activate stsim
python -m pip install -U pip
python -m pip install -e '.[full]'
```

## 2. Download the data

Download the project data from Google Drive:

https://drive.google.com/drive/folders/1Dme_UzJi1wTF_aiJrZSEfhtbzvn1SRbm?usp=sharing

Place the downloaded training data under the project `data/` directory, for example:

```text
data/
└── adata_backup.h5ad
```

## 3. Download Allen CCFv3

```bash
python scripts/download_ccfv3.py --out data/ccfv3
```

Expected files:

```text
data/ccfv3/
├── annotation_10.nrrd
└── structure_tree.json
```

## 4. Register the training data to CCFv3

```bash
python scripts/register_20260831_to_ccf.py \
  data/adata_backup.h5ad \
  --output data/adata_backup_ccf_hy.h5ad
```

The registered AnnData contains:

```text
obsm['ccf_coordinates']
obs['ccf_structure_id_native']
obs['ccf_domain']
obs['ccf_in_roi']
```

If needed, rerun registration with:

```bash
--ml-flip
```

and/or:

```bash
--dv-flip
```

## 5. Train the control model and simulate 3D tissues

```bash
python examples/full_ccfv3_hy_mvp.py \
  --control data/adata_backup.h5ad \
  --ccf-dir data/ccfv3 \
  --sample-size 3 \
  --work-dir runs/hy_control
```

For each simulated individual, the output contains:

```text
biological_tissue.npz
section_000.npz
section_001.npz
...
section_011.npz
```

To reuse an existing fitted model:

```bash
python examples/full_ccfv3_hy_mvp.py \
  --control data/adata_backup.h5ad \
  --ccf-dir data/ccfv3 \
  --sample-size 3 \
  --work-dir runs/hy_control \
  --reuse-fitted-model
```

## 6. Optional case-control simulation

```bash
python examples/full_ccfv3_hy_mvp.py \
  --control /path/to/control.h5ad \
  --disease /path/to/disease.h5ad \
  --ccf-dir data/ccfv3 \
  --sample-size 3 \
  --work-dir runs/hy_case_control
```

## 7. Run the tests

```bash
pytest -q
```
