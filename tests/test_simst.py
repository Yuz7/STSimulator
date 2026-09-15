from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simst import (
    CCFv3Atlas,
    CCFv3CanonicalBuilder,
    CCFv3Config,
    CellTypeModule,
    CellTypeOutput,
    DifferenceIntervention,
    DomainModule,
    DomainOutput,
    FittedSimST,
    GeneExpressionModule,
    GeneExpressionOutput,
    GeometryConfig,
    ReferenceData,
    SectionConfig,
    SimST,
    SimulationConfig,
)
from simst.synthetic import make_reference
from simst.geometry import GeometryModel
from simst.canonical import build_canonical_anatomy
from simst.cell_type import CellTypeMLPConfig, SpatialCellTypeMLP, SpatialProbMLP
from simst.domain import DomainMLPConfig, SpatialDomainMLP
from simst.expression import SpatialNBConfig, SpatialNBGeneExpression


class ReferenceDomain(DomainModule):
    def fit(self, reference, canonical):
        self.names = canonical.domain_names
        lookup = {name: i for i, name in enumerate(self.names)}
        self.index = np.array([lookup[x] for x in reference.domain_labels])
        return self

    def sample(self, geometry, rng):
        index = self.index[geometry.truth.source_indices]
        return DomainOutput(np.eye(len(self.names))[index], index, self.names)


class ReferenceCellType(CellTypeModule):
    def fit(self, reference, canonical):
        self.names = canonical.cell_type_names
        lookup = {name: i for i, name in enumerate(self.names)}
        self.index = np.array([lookup[x] for x in reference.cell_type_labels])
        return self

    def sample(self, geometry, domain, rng):
        index = self.index[geometry.truth.source_indices]
        return CellTypeOutput(np.eye(len(self.names))[index], index, self.names)


class ReferenceExpression(GeneExpressionModule):
    def fit(self, reference, canonical):
        self.value = reference.expression
        self.gene_names = reference.gene_names
        return self

    def sample(self, geometry, domain, cell_type, rng):
        value = self.value[geometry.truth.source_indices]
        return GeneExpressionOutput(value, value.copy(), self.gene_names)


class ConstantExpression(GeneExpressionModule):
    def fit(self, reference, canonical):
        self.gene_names = reference.gene_names
        return self

    def sample(self, geometry, domain, cell_type, rng):
        value = np.ones((len(geometry.coordinates), len(self.gene_names)))
        return GeneExpressionOutput(value, value.copy(), self.gene_names)


class SimSTMVPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control_reference = make_reference(200, 8, seed=19)
        cls.difference = np.arange(8, dtype=float) / 4
        cls.disease_reference = ReferenceData(
            coordinates=cls.control_reference.coordinates,
            domain_labels=cls.control_reference.domain_labels,
            cell_type_labels=cls.control_reference.cell_type_labels,
            expression=cls.control_reference.expression + cls.difference,
            gene_names=cls.control_reference.gene_names,
        )
        cls.engine = SimST(ReferenceDomain(), ReferenceCellType(), ReferenceExpression())
        cls.model = cls.engine.fit(cls.control_reference)
        cls.config = SimulationConfig(
            root_seed=73,
            geometry=GeometryConfig(n_points=180),
            sections=SectionConfig(n_sections=4, thickness=0.2, translation_sd=0.01),
        )

    def test_case_control_sample_size_and_matching(self):
        intervention = DifferenceIntervention.from_references(
            self.control_reference, self.disease_reference
        )
        dataset = self.engine.simulate_case_control(
            self.model, sample_size=3, intervention=intervention, config=self.config
        )
        self.assertEqual(len(dataset.control), 3)
        self.assertEqual(len(dataset.disease), 3)
        self.assertEqual(dataset.sample_size, 3)
        for control, disease in zip(dataset.control, dataset.disease):
            self.assertTrue(
                np.array_equal(control.tissue.geometry.coordinates, disease.tissue.geometry.coordinates)
            )
            self.assertTrue(np.array_equal(control.tissue.domain_index, disease.tissue.domain_index))
            self.assertTrue(
                np.array_equal(control.tissue.cell_type_index, disease.tissue.cell_type_index)
            )
            observed = disease.tissue.expression - control.tissue.expression
            self.assertTrue(np.allclose(observed, self.difference))
            self.assertEqual(len(control.sections), len(disease.sections))

    def test_modules_are_replaceable(self):
        engine = SimST(ReferenceDomain(), ReferenceCellType(), ConstantExpression())
        model = engine.fit(self.control_reference)
        sample = engine.simulate_one(model, self.config, sample_index=0)
        self.assertTrue(np.array_equal(sample.tissue.expression, np.ones((180, 8))))

    def test_reproducible_controls(self):
        first = self.engine.simulate_controls(self.model, 2, self.config)
        second = self.engine.simulate_controls(self.model, 2, self.config)
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left.tissue.expression, right.tissue.expression))
            self.assertTrue(
                np.array_equal(left.tissue.geometry.coordinates, right.tissue.geometry.coordinates)
            )

    def test_model_and_dataset_save(self):
        intervention = DifferenceIntervention.from_references(
            self.control_reference, self.disease_reference
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.model.save(root / "model")
            loaded = FittedSimST.load(root / "model")
            dataset = self.engine.simulate_case_control(
                loaded, sample_size=2, intervention=intervention, config=self.config
            )
            dataset.save(root / "run")
            self.assertTrue((root / "run/control/sample_0000/biological_tissue.npz").exists())
            self.assertTrue((root / "run/disease/sample_0001/biological_tissue.npz").exists())


    def test_spatial_celltype_model_is_pickle_safe(self):
        module = SpatialCellTypeMLP()
        module.names = np.asarray(["A", "B"], dtype=str)
        module.domain_names = np.asarray(["HY"], dtype=str)
        module.input_dim = 4
        module.model = SpatialProbMLP(4, 2).eval()
        restored = pickle.loads(pickle.dumps(module, protocol=pickle.HIGHEST_PROTOCOL))
        self.assertIsNotNone(restored.model)
        self.assertEqual(restored.input_dim, 4)

    def test_notebook_domain_mlp_fits_and_samples(self):
        module = SpatialDomainMLP(
            DomainMLPConfig(epochs=1, batch_size=64, device="cpu")
        )
        canonical = build_canonical_anatomy(self.control_reference)
        module.fit(self.control_reference, canonical)
        geometry = GeometryModel(canonical).sample(
            "sample",
            GeometryConfig(n_points=120),
            np.random.default_rng(5),
        )
        output = module.sample(geometry, np.random.default_rng(6))
        self.assertEqual(output.probability.shape[0], len(geometry.coordinates))
        self.assertTrue(np.allclose(output.probability.sum(axis=1), 1.0))

    def test_notebook_slice_radius_celltype_target(self):
        reference = ReferenceData(
            coordinates=self.control_reference.coordinates,
            domain_labels=self.control_reference.domain_labels,
            cell_type_labels=self.control_reference.cell_type_labels,
            expression=self.control_reference.expression,
            gene_names=self.control_reference.gene_names,
            section_id=np.asarray(["A"] * 100 + ["B"] * 100),
        )
        module = SpatialCellTypeMLP(
            CellTypeMLPConfig(
                neighborhood_mode="slice_radius",
                slice_neighbors=5,
                epochs=1,
                batch_size=64,
                device="cpu",
            )
        )
        module.fit(reference, build_canonical_anatomy(reference))
        self.assertEqual(module.neighborhood_diagnostics["mode"], "slice_radius")
        self.assertEqual(set(module.neighborhood_diagnostics["radius_by_section"]), {"A", "B"})

    def test_notebook_spatial_nb_fits_and_samples(self):
        rng = np.random.default_rng(31)
        counts = rng.poisson(lam=4.0, size=(200, 4)).astype(float)
        reference = ReferenceData(
            coordinates=self.control_reference.coordinates,
            domain_labels=self.control_reference.domain_labels,
            cell_type_labels=self.control_reference.cell_type_labels,
            expression=counts,
            gene_names=np.asarray(["g0", "g1", "g2", "g3"]),
            library_size=counts.sum(axis=1) + 1.0,
        )
        canonical = build_canonical_anatomy(reference)
        module = SpatialNBGeneExpression(
            SpatialNBConfig(
                n_spatial_basis=3,
                min_nonzero=2,
                poisson_maxiter=20,
                nb_maxiter=20,
            )
        ).fit(reference, canonical)
        geometry = GeometryModel(canonical).sample(
            "sample", GeometryConfig(n_points=80), np.random.default_rng(32)
        )
        source = geometry.truth.source_indices
        domain_names = np.unique(reference.domain_labels)
        domain_lookup = {name: i for i, name in enumerate(domain_names)}
        domain_index = np.asarray([domain_lookup[x] for x in reference.domain_labels[source]])
        domain = DomainOutput(
            np.eye(len(domain_names))[domain_index], domain_index, domain_names
        )
        cell_names = np.unique(reference.cell_type_labels)
        cell_lookup = {name: i for i, name in enumerate(cell_names)}
        cell_index = np.asarray([cell_lookup[x] for x in reference.cell_type_labels[source]])
        cell_type = CellTypeOutput(
            np.eye(len(cell_names))[cell_index], cell_index, cell_names
        )
        output = module.sample(geometry, domain, cell_type, np.random.default_rng(33))
        self.assertEqual(output.value.shape, (80, 4))
        self.assertTrue((output.value >= 0).all())
        self.assertEqual(set(module.fit_status.values()), {"success"})


    def test_ccf_canonical_coordinates_stay_inside_roi_after_individual_jitter(self):
        # Regression test for v0.3.2: individual coordinate jitter must alter only
        # the physical realization, never the canonical CCF coordinate used for
        # deterministic domain lookup.
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = np.zeros((8, 8, 8), dtype=np.int32)
            annotation[2:6, 2:6, 2:6] = 1097
            np.save(root / "annotation.npy", annotation)
            structure_tree = [
                {"id": 997, "acronym": "root", "name": "root", "parent_structure_id": None},
                {"id": 1097, "acronym": "HY", "name": "Hypothalamus", "parent_structure_id": 997},
            ]
            (root / "structures.json").write_text(json.dumps(structure_tree), encoding="utf-8")

            atlas = CCFv3Atlas(
                CCFv3Config(
                    annotation_path=root / "annotation.npy",
                    structure_tree_path=root / "structures.json",
                    roi_structure_ids=(1097,),
                    ccf_voxel_resolution=10,
                    domain_granularity="roi_root",
                )
            )
            reference = make_reference(20, 3, seed=11)
            canonical = CCFv3CanonicalBuilder(atlas).build(reference)
            geometry = GeometryModel(canonical).sample(
                "sample",
                GeometryConfig(
                    n_points=200,
                    coordinate_jitter=0.8,
                    ccf_subvoxel_jitter=True,
                    scale_sd=0.0,
                    anisotropy_sd=0.0,
                    rotation_sd_degrees=0.0,
                    warp_amplitude=0.0,
                    boundary_strength=0.0,
                ),
                np.random.default_rng(7),
            )
            labels = atlas.domain_labels(geometry.canonical_coordinates)
            self.assertEqual(set(labels.tolist()), {"HY"})
            # Physical coordinates are allowed to leave the canonical ROI after
            # individual-specific perturbation/deformation.
            self.assertFalse(np.array_equal(geometry.coordinates, geometry.canonical_coordinates))

    def test_invalid_sample_size_fails(self):
        with self.assertRaises(ValueError):
            self.engine.simulate_controls(self.model, 0, self.config)


if __name__ == "__main__":
    unittest.main()
