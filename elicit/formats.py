"""Three report formats obtained by projecting one quadratic true surplus."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elicit.utilities import QuadraticSurplusModel


@dataclass(frozen=True)
class FullReport:
    """Quadratic report parameterised by peak, upper zero, and scale.

    The normalised rule divides out ``scale`` and gives every actor unit
    reported surplus at its peak. This is the explicit anti-hiking restriction:
    scale remains recorded, but cannot buy influence.
    """

    peak: float
    upper: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.peak < 0 or self.upper <= self.peak:
            raise ValueError("full report requires 0 <= peak < upper")
        if self.scale <= 0:
            raise ValueError("full report scale must be positive")

    @property
    def half_width(self) -> float:
        return self.upper - self.peak

    @property
    def lower(self) -> float:
        return max(0.0, 2.0 * self.peak - self.upper)

    def value(self, price, *, normalise: bool = True):
        p = np.asarray(price, dtype=float)
        amplitude = 1.0 if normalise else self.scale
        value = amplitude * (1.0 - ((p - self.peak) / self.half_width) ** 2)
        return float(value) if value.ndim == 0 else value


@dataclass(frozen=True)
class PeakReport:
    """A single preferred non-negative price."""

    peak: float

    def __post_init__(self) -> None:
        if self.peak < 0:
            raise ValueError("peak report must be non-negative")


@dataclass(frozen=True)
class RegionReport:
    """Closed interval of reported non-negative utility."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower < 0 or self.upper < self.lower:
            raise ValueError("region report requires 0 <= lower <= upper")

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    def accepts(self, price: float, tolerance: float = 1e-12) -> bool:
        return self.lower - tolerance <= price <= self.upper + tolerance


def truthful_full(model: QuadraticSurplusModel) -> list[FullReport]:
    return [
        FullReport(model.peak(i), model.region(i)[1], 1.0)
        for i in range(len(model.names))
    ]


def truthful_peaks(model: QuadraticSurplusModel) -> list[PeakReport]:
    return [PeakReport(model.peak(i)) for i in range(len(model.names))]


def truthful_regions(model: QuadraticSurplusModel) -> list[RegionReport]:
    return [RegionReport(*model.region(i)) for i in range(len(model.names))]

