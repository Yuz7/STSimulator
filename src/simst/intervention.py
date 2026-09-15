"""Pluggable interventions applied to a generated control tissue."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from .data import BiologicalTissue, ReferenceData


class InterventionModule(ABC):
    @abstractmethod
    def apply(
        self,
        control: BiologicalTissue,
        rng: np.random.Generator,
    ) -> BiologicalTissue:
        pass


class DifferenceIntervention(InterventionModule):
    """Add a gene-wise disease-minus-control mean difference to every location."""

    def __init__(self, gene_names: np.ndarray, difference: np.ndarray):
        self.gene_names = np.asarray(gene_names, dtype=str)
        self.difference = np.asarray(difference, dtype=float)

    @classmethod
    def from_references(
        cls,
        control: ReferenceData,
        disease: ReferenceData,
    ) -> "DifferenceIntervention":
        if not np.array_equal(control.gene_names, disease.gene_names):
            raise ValueError("Control and disease references must have identical genes in identical order")
        difference = disease.expression.mean(axis=0) - control.expression.mean(axis=0)
        return cls(control.gene_names, difference)

    def apply(
        self,
        control: BiologicalTissue,
        rng: np.random.Generator,
    ) -> BiologicalTissue:
        return replace(
            control,
            expression_mean=np.maximum(control.expression_mean + self.difference, 0.0),
            expression=np.maximum(control.expression + self.difference, 0.0),
        )
