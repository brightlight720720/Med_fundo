"""Composite reward formula recorded in the submitted Methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    diagnosis: float = 0.5
    format: float = 0.1
    explanation_diagnosis: float = 0.4

    def validate(self) -> None:
        total = self.diagnosis + self.format + self.explanation_diagnosis
        if abs(total - 1.0) > 1e-12:
            raise ValueError("Reward weights must sum to 1.0")


def composite_reward(
    diagnosis: float,
    format_score: float,
    explanation_diagnosis: float,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """Combine already normalized component scores.

    The exact historical component implementation was not recovered. This
    function therefore does not infer or manufacture those components.
    """
    weights.validate()
    values = (diagnosis, format_score, explanation_diagnosis)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Each reward component must be normalized to [0, 1]")
    return (
        weights.diagnosis * diagnosis
        + weights.format * format_score
        + weights.explanation_diagnosis * explanation_diagnosis
    )
