"""Model-agnostic simST orchestration."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .canonical import build_canonical_anatomy
from .cell_type import CellTypeModule
from .config import SimulationConfig
from .data import (
    BiologicalTissue,
    CanonicalAnatomy,
    CaseControlDataset,
    IdealSection,
    ReferenceData,
    SimulatedIndividual,
)
from .domain import DomainModule
from .expression import GeneExpressionModule
from .geometry import GeometryModel
from .intervention import InterventionModule
from .random import (
    CELL_TYPE_STREAM,
    DOMAIN_STREAM,
    EXPRESSION_STREAM,
    GEOMETRY_STREAM,
    INTERVENTION_STREAM,
    SECTION_STREAM,
    SeedTree,
)
from .section import SectionSampler


class CanonicalBuilder(Protocol):
    def build(self, reference: ReferenceData) -> CanonicalAnatomy:
        ...


@dataclass
class FittedSimST:
    canonical: CanonicalAnatomy
    geometry_model: GeometryModel
    domain_module: DomainModule
    cell_type_module: CellTypeModule
    gene_expression_module: GeneExpressionModule

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "model.pkl"
        temporary = directory / "model.pkl.tmp"
        try:
            with temporary.open("wb") as stream:
                pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, directory: str | Path) -> "FittedSimST":
        with (Path(directory) / "model.pkl").open("rb") as stream:
            return pickle.load(stream)


class SimST:
    def __init__(
        self,
        domain_module: DomainModule,
        cell_type_module: CellTypeModule,
        gene_expression_module: GeneExpressionModule,
        canonical_builder: CanonicalBuilder | None = None,
    ):
        self.domain_module = domain_module
        self.cell_type_module = cell_type_module
        self.gene_expression_module = gene_expression_module
        self.canonical_builder = canonical_builder

    def fit(self, reference: ReferenceData) -> FittedSimST:
        if self.canonical_builder is None:
            canonical = build_canonical_anatomy(reference)
        else:
            canonical = self.canonical_builder.build(reference)

        fitted_domain = self.domain_module.fit(reference, canonical)
        # CCFv3 modules replace placeholder domains here; legacy modules simply
        # return the original ReferenceData unchanged.
        training_reference = fitted_domain.annotate_reference(reference, canonical)

        return FittedSimST(
            canonical=canonical,
            geometry_model=GeometryModel(canonical),
            domain_module=fitted_domain,
            cell_type_module=self.cell_type_module.fit(training_reference, canonical),
            gene_expression_module=self.gene_expression_module.fit(
                training_reference, canonical
            ),
        )

    def simulate_one(
        self,
        model: FittedSimST,
        config: SimulationConfig,
        sample_index: int,
    ) -> SimulatedIndividual:
        seeds = SeedTree(config.root_seed)
        geometry = model.geometry_model.sample(
            f"sample_{sample_index:04d}",
            config.geometry,
            seeds.rng(sample_index, GEOMETRY_STREAM),
        )
        domain = model.domain_module.sample(
            geometry,
            seeds.rng(sample_index, DOMAIN_STREAM),
        )
        cell_type = model.cell_type_module.sample(
            geometry,
            domain,
            seeds.rng(sample_index, CELL_TYPE_STREAM),
        )
        gene_expression = model.gene_expression_module.sample(
            geometry,
            domain,
            cell_type,
            seeds.rng(sample_index, EXPRESSION_STREAM),
        )
        tissue = BiologicalTissue(
            geometry=geometry,
            domain_probability=domain.probability,
            domain_index=domain.index,
            domain_names=domain.names,
            cell_type_probability=cell_type.probability,
            cell_type_index=cell_type.index,
            cell_type_names=cell_type.names,
            expression_mean=gene_expression.mean,
            expression=gene_expression.value,
            gene_names=gene_expression.gene_names,
        )
        sections = self._make_sections(tissue, config, sample_index)
        return SimulatedIndividual(tissue=tissue, sections=sections)

    def simulate_controls(
        self,
        model: FittedSimST,
        sample_size: int,
        config: SimulationConfig,
    ) -> tuple[SimulatedIndividual, ...]:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        return tuple(self.simulate_one(model, config, i) for i in range(sample_size))

    def simulate_case_control(
        self,
        model: FittedSimST,
        sample_size: int,
        intervention: InterventionModule,
        config: SimulationConfig,
    ) -> CaseControlDataset:
        control = self.simulate_controls(model, sample_size, config)
        seeds = SeedTree(config.root_seed)
        disease = tuple(
            self._intervene(
                individual,
                intervention,
                seeds.rng(i, INTERVENTION_STREAM),
            )
            for i, individual in enumerate(control)
        )
        return CaseControlDataset(control=control, disease=disease)

    def _make_sections(
        self,
        tissue: BiologicalTissue,
        config: SimulationConfig,
        sample_index: int,
    ) -> tuple[IdealSection, ...]:
        if config.sections is None:
            return ()
        return SectionSampler().sample(
            tissue,
            config.sections,
            SeedTree(config.root_seed).rng(sample_index, SECTION_STREAM),
        )

    @staticmethod
    def _intervene(
        control: SimulatedIndividual,
        intervention: InterventionModule,
        rng,
    ) -> SimulatedIndividual:
        tissue = intervention.apply(control.tissue, rng)
        sections = tuple(
            IdealSection(
                section_id=section.section_id,
                source_indices=section.source_indices,
                coordinates_2d=section.coordinates_2d,
                plane_origin=section.plane_origin,
                plane_normal=section.plane_normal,
                expression=tissue.expression[section.source_indices],
                domain_index=tissue.domain_index[section.source_indices],
                cell_type_index=tissue.cell_type_index[section.source_indices],
            )
            for section in control.sections
        )
        return SimulatedIndividual(tissue=tissue, sections=sections)
