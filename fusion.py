"""Opinion fusion for three calibrated supports.

This subset accepts e_D, e_Q and e_H after normality calibration. It omits
image evidence extraction, calibration, gates and normal state stabilization;
it does not reproduce the manuscript's experiments or complete NCTM pipeline.
Trust is the harmonic mean of supports. Conflict is a separate diagnostic and
does not alter trust, uncertainty or fusion weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import fsum, isclose, isfinite
from numbers import Real
from typing import Iterable


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1].") from exc
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return number


@dataclass(frozen=True)
class Opinion:
    """Belief, disbelief, uncertainty and base rate for a binary state."""

    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float = 0.5

    def __post_init__(self) -> None:
        for name in ("belief", "disbelief", "uncertainty", "base_rate"):
            _unit_interval(getattr(self, name), name)
        if not isclose(
            fsum((self.belief, self.disbelief, self.uncertainty)),
            1.0, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("Belief, disbelief and uncertainty must sum to 1.")

    @property
    def projected_probability(self) -> float:
        """Opinion projection; this is distinct from the reported trust."""
        return self.belief + self.base_rate * self.uncertainty


@dataclass(frozen=True)
class FusionResult:
    trust: float
    uncertainty: float
    conflict: float
    fused_opinion: Opinion
    source_opinions: tuple[Opinion, ...]


def fuse_supports(
    supports: Iterable[float],
    min_uncertainty: float = 0.05,
    max_uncertainty: float = 0.35,
) -> FusionResult:
    """Fuse exactly three calibrated supports in the order e_D, e_Q, e_H.

    Supports must be finite numbers in [0, 1]; uncertainty bounds must satisfy
    0 < min_uncertainty <= max_uncertainty <= 1. A zero support gives zero trust
    by the harmonic mean limit. All outputs are dimensionless.
    """
    try:
        values = tuple(supports)
    except TypeError as exc:
        raise ValueError("Provide exactly three calibrated supports.") from exc
    if len(values) != 3:
        raise ValueError("Provide exactly three calibrated supports.")
    values = tuple(_unit_interval(value, "support") for value in values)
    lower = _unit_interval(min_uncertainty, "min_uncertainty")
    upper = _unit_interval(max_uncertainty, "max_uncertainty")
    if not 0.0 < lower <= upper:
        raise ValueError("Uncertainty bounds must satisfy 0 < min <= max <= 1.")

    opinions = []
    for support in values:
        uncertainty = lower + (upper - lower) * 4.0 * support * (1.0 - support)
        committed = 1.0 - uncertainty
        opinions.append(Opinion(
            committed * support, committed * (1.0 - support), uncertainty,
        ))

    # Scale inverse uncertainties to avoid overflow for tiny positive bounds.
    scale = min(opinion.uncertainty for opinion in opinions)
    inverse = tuple(scale / opinion.uncertainty for opinion in opinions)
    denominator = fsum(inverse)
    weights = tuple(value / denominator for value in inverse)
    uncertainty = scale * (3.0 / denominator)
    fused = Opinion(
        fsum(weight * item.belief for weight, item in zip(weights, opinions)),
        fsum(weight * item.disbelief for weight, item in zip(weights, opinions)),
        uncertainty,
    )
    conflict = fsum(
        left.belief * right.disbelief + left.disbelief * right.belief
        for left, right in combinations(opinions, 2)
    ) / 3.0

    # The scaled harmonic mean also works for positive subnormal supports.
    smallest = min(values)
    trust = 0.0 if smallest == 0.0 else (
        smallest * (3.0 / fsum(smallest / value for value in values))
    )
    return FusionResult(trust, uncertainty, conflict, fused, tuple(opinions))
