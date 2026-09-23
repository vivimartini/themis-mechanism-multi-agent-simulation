"""Calibrated peaked price surpluses for the report-format experiment.

With transfers off and coverage fixed at ``c``, the existing affine preference
layer supplies an upper acceptable price

    h_i(c) = alpha_base_i + alpha_cov_i * c.

The new scenario parameter ``peak_fraction`` places the utility maximum at
``mu_i = peak_fraction * h_i``.  The quadratic is normalised to one at the peak
and anchored to zero at the existing threshold:

    s_i(p) = 1 - ((p - mu_i) / (h_i - mu_i))**2.

Its other zero is ``2*mu_i - h_i``.  Prices are non-negative, so the reported
acceptable region clips that lower zero at zero.  No data identify the peak
fraction; callers must therefore report or sweep it explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from paths import ACTORS_CSV

DEFAULT_COVERAGE = 0.8686437102493683
DEFAULT_PEAK_FRACTIONS = (0.35, 0.50, 0.65)


@dataclass(frozen=True)
class QuadraticSurplusModel:
    """One peaked true surplus per actor at a fixed coverage scenario."""

    names: tuple[str, ...]
    weights: np.ndarray
    alpha_base: np.ndarray
    alpha_cov: np.ndarray
    coverage: float
    peak_fraction: float | np.ndarray

    def __post_init__(self) -> None:
        n = len(self.names)
        arrays = (self.weights, self.alpha_base, self.alpha_cov)
        if any(np.asarray(a).shape != (n,) for a in arrays):
            raise ValueError("actor arrays must be one-dimensional and equally sized")
        fractions = np.asarray(self.peak_fraction, dtype=float)
        if fractions.ndim > 1 or (fractions.ndim == 1 and fractions.shape != (n,)):
            raise ValueError("peak_fraction must be scalar or one value per actor")
        if np.any((fractions <= 0.0) | (fractions >= 1.0)):
            raise ValueError("peak fractions must lie strictly between zero and one")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must lie in [0, 1]")
        if np.any(self.thresholds <= 0):
            raise ValueError("all actors need a positive upper acceptable price")
        if np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise ValueError("weights must be non-negative with positive total")

    @property
    def thresholds(self) -> np.ndarray:
        """Existing transfer-free affine willingness at the scenario coverage."""
        return self.alpha_base + self.alpha_cov * self.coverage

    @property
    def peaks(self) -> np.ndarray:
        return self.peak_fractions * self.thresholds

    @property
    def peak_fractions(self) -> np.ndarray:
        fractions = np.asarray(self.peak_fraction, dtype=float)
        if fractions.ndim == 0:
            return np.full(len(self.names), float(fractions))
        return fractions

    @property
    def half_widths(self) -> np.ndarray:
        return self.thresholds - self.peaks

    def peak(self, i: int) -> float:
        return float(self.peaks[i])

    def region(self, i: int) -> tuple[float, float]:
        """Non-negative-price region where actor ``i`` has non-negative surplus."""
        lower = max(0.0, 2.0 * self.peak(i) - float(self.thresholds[i]))
        return lower, float(self.thresholds[i])

    def surplus(self, i: int, price):
        """True cardinal surplus relative to the zero outside option."""
        p = np.asarray(price, dtype=float)
        value = 1.0 - ((p - self.peaks[i]) / self.half_widths[i]) ** 2
        return float(value) if value.ndim == 0 else value

    def dichotomous_surplus(self, i: int, price) -> float:
        """Approval utility: one inside the true region and zero otherwise."""
        lo, hi = self.region(i)
        return float(lo - 1e-12 <= float(price) <= hi + 1e-12)

    def subset(self, indices: Iterable[int]) -> "QuadraticSurplusModel":
        idx = np.asarray(tuple(indices), dtype=int)
        weights = self.weights[idx].copy()
        weights /= weights.sum()
        return QuadraticSurplusModel(
            names=tuple(self.names[i] for i in idx),
            weights=weights,
            alpha_base=self.alpha_base[idx].copy(),
            alpha_cov=self.alpha_cov[idx].copy(),
            coverage=self.coverage,
            peak_fraction=self.peak_fractions[idx].copy(),
        )


def load_model(
    coverage: float = DEFAULT_COVERAGE,
    peak_fraction: float | Iterable[float] = 0.50,
    actors_csv: str | Path = ACTORS_CSV,
) -> QuadraticSurplusModel:
    """Build the transfer-free surplus model from the dissertation calibration."""
    frame = pd.read_csv(actors_csv)
    emissions = frame["pop_m"].to_numpy(float) * frame["e"].to_numpy(float)
    weights = emissions / emissions.sum()
    return QuadraticSurplusModel(
        names=tuple(frame["name"].astype(str)),
        weights=weights,
        alpha_base=frame["alpha_base"].to_numpy(float),
        alpha_cov=frame["alpha_cov"].to_numpy(float),
        coverage=float(coverage),
        peak_fraction=(
            float(peak_fraction)
            if np.asarray(peak_fraction).ndim == 0
            else np.asarray(tuple(peak_fraction), dtype=float)
        ),
    )

