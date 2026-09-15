"""Configuration for the model-agnostic MVP simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeometryConfig:
    n_points: int | None = None
    scale_sd: float = 0.06
    anisotropy_sd: float = 0.03
    rotation_sd_degrees: float = 3.0
    warp_amplitude: float = 0.04
    warp_length_scale: float = 0.35
    boundary_strength: float = 0.03
    coordinate_jitter: float = 0.005
    ccf_subvoxel_jitter: bool = True
    random_features: int = 24


@dataclass(frozen=True)
class SectionConfig:
    n_sections: int = 12
    # Coordinates are in micrometres in the CCF pipeline.
    thickness: float = 10.0
    rotation_sd_degrees: float = 1.0
    translation_sd: float = 5.0
    adjacent_correlation: float = 0.8
    min_points: int = 5
    # MVP option to match the observed 12-section sample sizes.  When supplied,
    # the closest cells to each section plane are selected to exactly this count.
    # This is a calibration convenience, not a literal fixed-thickness section.
    target_counts: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SimulationConfig:
    root_seed: int = 0
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sections: SectionConfig | None = field(default_factory=SectionConfig)
