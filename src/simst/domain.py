"""Pluggable spatial-domain interface and notebook MLP implementation."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .data import CanonicalAnatomy, IndividualGeometry, ReferenceData


@dataclass(frozen=True)
class DomainOutput:
    probability: np.ndarray
    index: np.ndarray
    names: np.ndarray


class DomainModule(ABC):
    """Coordinates-to-domain component used by :class:`simst.model.SimST`."""

    @abstractmethod
    def fit(self, reference: ReferenceData, canonical: CanonicalAnatomy) -> "DomainModule":
        pass

    def annotate_reference(
        self,
        reference: ReferenceData,
        canonical: CanonicalAnatomy,
    ) -> ReferenceData:
        return reference

    @abstractmethod
    def sample(
        self,
        geometry: IndividualGeometry,
        rng: np.random.Generator,
    ) -> DomainOutput:
        pass


@dataclass(frozen=True)
class DomainMLPConfig:
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    split_seed: int = 42
    training_seed: int = 20260517
    device: str = "auto"


class SpatialDomainMLPNetwork(nn.Module):
    def __init__(self, input_dim: int, n_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_domains),
        )

    def forward(self, x):
        return self.net(x)


class SpatialDomainMLP(DomainModule):
    """Learn domain labels from registered three-dimensional coordinates."""

    def __init__(self, config: DomainMLPConfig | None = None):
        self.config = config or DomainMLPConfig()
        self.scaler = None
        self.label_encoder = None
        self.model: SpatialDomainMLPNetwork | None = None
        self.names: np.ndarray | None = None
        self.validation_accuracy: float | None = None

    def fit(self, reference: ReferenceData, canonical: CanonicalAnatomy) -> "SpatialDomainMLP":
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from torch.utils.data import DataLoader, TensorDataset

        coordinates = np.asarray(reference.ccf_coordinates, dtype=float)
        labels = np.asarray(reference.domain_labels, dtype=str)

        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(coordinates)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.names = np.asarray(self.label_encoder.classes_, dtype=str)

        x_train, x_validation, y_train, y_validation = train_test_split(
            x,
            y,
            test_size=self.config.validation_fraction,
            stratify=y,
            random_state=self.config.split_seed,
        )

        random.seed(self.config.training_seed)
        np.random.seed(self.config.training_seed)
        torch.manual_seed(self.config.training_seed)

        device = self._device()
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        model = SpatialDomainMLPNetwork(x.shape[1], len(self.names)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        for _ in range(self.config.epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(
                torch.tensor(x_validation, dtype=torch.float32, device=device)
            )
            validation_prediction = torch.argmax(validation_logits, dim=1).cpu().numpy()
        self.validation_accuracy = float(np.mean(validation_prediction == y_validation))
        self.model = model.to("cpu").eval()
        return self

    def sample(
        self,
        geometry: IndividualGeometry,
        rng: np.random.Generator,
    ) -> DomainOutput:
        if self.model is None or self.scaler is None or self.names is None:
            raise RuntimeError("SpatialDomainMLP must be fitted before sampling")
        coordinates = self.scaler.transform(geometry.canonical_coordinates)
        with torch.no_grad():
            logits = self.model(torch.tensor(coordinates, dtype=torch.float32))
            probability = torch.softmax(logits, dim=1).cpu().numpy()
        index = np.argmax(probability, axis=1).astype(np.int64)
        return DomainOutput(probability=probability, index=index, names=self.names)

    def _device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
