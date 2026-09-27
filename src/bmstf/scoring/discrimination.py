"""Discrimination read-outs and the net-reclassification statistic.

Ref: Table 1 panel A (AUROC for threshold crossing within 12 months); Table 5
panel A (categorical net reclassification against maximum diameter at
pre-specified risk categories).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RISK_CATEGORY_EDGES: tuple[float, ...] = (0.10, 0.20, 0.35)
NET_BENEFIT_THRESHOLDS: tuple[float, ...] = (0.05, 0.30)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based area under the ROC curve with mid-ranks for ties."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    outcomes = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.shape != outcomes.shape:
        raise ValueError("scores and labels must share a shape")
    positive = outcomes == 1.0
    negative = ~positive
    positives = int(np.count_nonzero(positive))
    negatives = int(np.count_nonzero(negative))
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = _average_ranks(values)
    return float(
        (float(np.sum(ranks[positive])) - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    position = 0
    while position < sorted_values.shape[0]:
        end = position
        while (
            end + 1 < sorted_values.shape[0] and sorted_values[end + 1] == sorted_values[position]
        ):
            end += 1
        average = 0.5 * (position + end) + 1.0
        ranks[order[position : end + 1]] = average
        position = end + 1
    return ranks


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float
    youden: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold),
            "sensitivity": float(self.sensitivity),
            "specificity": float(self.specificity),
            "youden": float(self.youden),
        }


def operating_point(scores: np.ndarray, labels: np.ndarray) -> OperatingPoint:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    outcomes = np.asarray(labels, dtype=np.float64).reshape(-1)
    candidates = np.unique(values)
    best = OperatingPoint(
        threshold=float(candidates[0]), sensitivity=0.0, specificity=0.0, youden=-1.0
    )
    for threshold in candidates:
        predicted = values >= threshold
        positives = outcomes == 1.0
        negatives = ~positives
        sensitivity = _safe_ratio(
            int(np.count_nonzero(predicted & positives)), int(np.count_nonzero(positives))
        )
        specificity = _safe_ratio(
            int(np.count_nonzero(~predicted & negatives)), int(np.count_nonzero(negatives))
        )
        youden = sensitivity + specificity - 1.0
        if youden > best.youden:
            best = OperatingPoint(float(threshold), sensitivity, specificity, youden)
    return best


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0


def assign_risk_category(
    probability: np.ndarray, edges: tuple[float, ...] = RISK_CATEGORY_EDGES
) -> np.ndarray:
    return np.asarray(np.digitize(np.asarray(probability, dtype=np.float64), np.asarray(edges)))


@dataclass(frozen=True, slots=True)
class Reclassification:
    nri: float
    event_up: float
    event_down: float
    non_event_up: float
    non_event_down: float
    counts: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "nri": float(self.nri),
            "event_up": float(self.event_up),
            "event_down": float(self.event_down),
            "non_event_up": float(self.non_event_up),
            "non_event_down": float(self.non_event_down),
            "counts": float(self.counts),
        }


def categorical_nri(
    new_probability: np.ndarray,
    old_probability: np.ndarray,
    outcomes: np.ndarray,
    edges: tuple[float, ...] = RISK_CATEGORY_EDGES,
) -> Reclassification:
    """Categorical NRI with the standard category-shift definition."""

    new = assign_risk_category(new_probability, edges)
    old = assign_risk_category(old_probability, edges)
    labels = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    events = labels == 1.0
    non_events = ~events
    event_count = int(np.count_nonzero(events))
    non_event_count = int(np.count_nonzero(non_events))
    event_up = _safe_ratio(int(np.count_nonzero((new > old) & events)), event_count)
    event_down = _safe_ratio(int(np.count_nonzero((new < old) & events)), event_count)
    non_event_up = _safe_ratio(int(np.count_nonzero((new > old) & non_events)), non_event_count)
    non_event_down = _safe_ratio(int(np.count_nonzero((new < old) & non_events)), non_event_count)
    return Reclassification(
        nri=float(event_up - event_down + non_event_down - non_event_up),
        event_up=event_up,
        event_down=event_down,
        non_event_up=non_event_up,
        non_event_down=non_event_down,
        counts=int(labels.shape[0]),
    )


def net_benefit(
    sensitivity: float, specificity: float, prevalence: float, threshold: float
) -> float:
    """Net benefit of a decision rule at a probability threshold."""

    if threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    return float(
        float(sensitivity) * float(prevalence)
        - (1.0 - float(specificity))
        * (1.0 - float(prevalence))
        * (float(threshold) / (1.0 - float(threshold)))
    )


def treat_all_net_benefit(prevalence: float, threshold: float) -> float:
    if threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    return float(
        float(prevalence) - (1.0 - float(prevalence)) * float(threshold) / (1.0 - float(threshold))
    )
