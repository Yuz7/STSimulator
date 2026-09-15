import numpy as np

from simst import (
    CellTypeModule,
    CellTypeOutput,
    DifferenceIntervention,
    DomainModule,
    DomainOutput,
    GeneExpressionModule,
    GeneExpressionOutput,
    ReferenceData,
    SimST,
    SimulationConfig,
)
from simst.synthetic import make_reference


class ReferenceDomain(DomainModule):
    def fit(self, reference, canonical):
        self.names = canonical.domain_names
        lookup = {name: i for i, name in enumerate(self.names)}
        self.index = np.array([lookup[x] for x in reference.domain_labels])
        return self

    def sample(self, geometry, rng):
        index = self.index[geometry.truth.source_indices]
        probability = np.eye(len(self.names))[index]
        return DomainOutput(probability, index, self.names)


class ReferenceCellType(CellTypeModule):
    def fit(self, reference, canonical):
        self.names = canonical.cell_type_names
        lookup = {name: i for i, name in enumerate(self.names)}
        self.index = np.array([lookup[x] for x in reference.cell_type_labels])
        return self

    def sample(self, geometry, domain, rng):
        index = self.index[geometry.truth.source_indices]
        probability = np.eye(len(self.names))[index]
        return CellTypeOutput(probability, index, self.names)


class ReferenceExpression(GeneExpressionModule):
    def fit(self, reference, canonical):
        self.expression = reference.expression
        self.gene_names = reference.gene_names
        return self

    def sample(self, geometry, domain, cell_type, rng):
        value = self.expression[geometry.truth.source_indices]
        return GeneExpressionOutput(value, value.copy(), self.gene_names)


control_reference = make_reference(n_locations=1200, n_genes=24, seed=7)
disease_reference = ReferenceData(
    coordinates=control_reference.coordinates,
    domain_labels=control_reference.domain_labels,
    cell_type_labels=control_reference.cell_type_labels,
    expression=control_reference.expression + np.linspace(0.0, 2.0, 24),
    gene_names=control_reference.gene_names,
)

engine = SimST(ReferenceDomain(), ReferenceCellType(), ReferenceExpression())
model = engine.fit(control_reference)
intervention = DifferenceIntervention.from_references(control_reference, disease_reference)
dataset = engine.simulate_case_control(
    model=model,
    sample_size=3,
    intervention=intervention,
    config=SimulationConfig(root_seed=20260802),
)
dataset.save("runs/quickstart")

print(len(dataset.control), len(dataset.disease))
