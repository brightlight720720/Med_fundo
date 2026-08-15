"""Decision-curve helpers."""

from __future__ import annotations

import numpy as np


def net_benefit(y_true: np.ndarray, decisions: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true, dtype=int)
    decisions = np.asarray(decisions, dtype=int)
    if y_true.shape != decisions.shape:
        raise ValueError("y_true and decisions must have the same shape")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between 0 and 1")
    true_positive = np.sum((y_true == 1) & (decisions == 1))
    false_positive = np.sum((y_true == 0) & (decisions == 1))
    odds = threshold / (1.0 - threshold)
    return float(
        (true_positive / len(y_true)) - (false_positive / len(y_true)) * odds
    )


def refer_all_net_benefit(
    prevalence: float, threshold: np.ndarray
) -> np.ndarray:
    threshold = np.asarray(threshold, dtype=float)
    return prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
