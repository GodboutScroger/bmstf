"""Energy score: the proper score used to select the latent dimension.

Ref: Sec. "Pre-specified clinical protocol" (the energy score applied to the
validation folds was responsible for the selection of the latent dimension d,
between 8 and 16); Sec. "Two-regime latent shear-state model" (minimising the
score of the mixture forecast against the next observation).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm

LATENT_DIMENSION_RANGE: tuple[int, int] = (8, 16)


def energy_score_samples(samples: np.ndarray, observation: np.ndarray, beta: float = 1.0) -> float:
    """(1/M) sum ||x_i - y||^beta - 1/(2 M^2) sum ||x_i - x_j||^beta."""

    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(observation, dtype=np.float64).reshape(-1)
    count = values.shape[0]
    if count == 0:
        return 0.0
    first = float(np.mean(np.linalg.norm(values - target[None, :], axis=1) ** float(beta)))
    second = _pairwise_norm_mean(values, float(beta))
    return float(first - 0.5 * second)


def _pairwise_norm_mean(values: np.ndarray, beta: float, chunk: int = 128) -> float:
    """Mean pairwise distance, computed in row chunks so the cost stays O(M^2) but vectorised."""

    count = values.shape[0]
    if count == 0:
        return 0.0
    total = 0.0
    for start in range(0, count, int(chunk)):
        block = values[start : start + int(chunk)]
        difference = block[:, None, :] - values[None, :, :]
        total += float(np.sum(np.linalg.norm(difference, axis=2) ** beta))
    return total / float(count**2)


def expected_norm(mean: np.ndarray, covariance: np.ndarray) -> float:
    """E||X|| for X ~ N(mean, covariance) by exact polar quadrature in two dimensions."""

    centre = np.asarray(mean, dtype=np.float64).reshape(-1)
    matrix = np.asarray(covariance, dtype=np.float64)
    return _gaussian_norm_expectation(centre, matrix, np.zeros_like(centre))


def energy_score_gaussian(
    mean: np.ndarray, covariance: np.ndarray, observation: np.ndarray
) -> float:
    """E||X - y|| - 1/2 E||X - X'|| for a Gaussian forecast, exact in two dimensions."""

    centre = np.asarray(mean, dtype=np.float64).reshape(-1)
    target = np.asarray(observation, dtype=np.float64).reshape(-1)
    matrix = np.asarray(covariance, dtype=np.float64)
    first = _gaussian_norm_expectation(centre, matrix, target)
    second = _gaussian_norm_expectation(centre, 2.0 * matrix, centre)
    return float(first - 0.5 * second)


def _gaussian_norm_expectation(
    mean: np.ndarray, covariance: np.ndarray, target: np.ndarray
) -> float:
    """E||X - target|| for a two-dimensional Gaussian, by polar quadrature.

    Writing the difference in the eigenbasis of the covariance, the expectation is
    the double integral of r^2 times the two marginal densities over the plane.
    """

    centre = np.asarray(mean, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if centre.shape[0] != 2:
        raise ValueError("the exact Gaussian reference is implemented for two dimensions")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    scale = np.sqrt(np.clip(values, 1.0e-12, None))
    projection = vectors.T @ centre

    def integrand(radius: float, angle: float) -> float:
        x = radius * float(np.cos(angle))
        y = radius * float(np.sin(angle))
        density = float(
            norm.pdf((x - projection[0]) / scale[0])
            * norm.pdf((y - projection[1]) / scale[1])
            / (scale[0] * scale[1])
        )
        return radius * radius * density

    value, _ = dblquad(integrand, 0.0, 2.0 * np.pi, lambda _: 0.0, lambda _: 16.0, epsabs=1.0e-11)
    return float(value)


@dataclass(frozen=True, slots=True)
class DimensionSelection:
    dimension: int
    scores: dict[int, float]

    def as_mapping(self) -> dict[str, float]:
        mapping = {f"score_d{key}": float(value) for key, value in self.scores.items()}
        mapping["selected"] = float(self.dimension)
        return mapping


def select_latent_dimension(
    candidates: Sequence[int],
    evaluate: Callable[[int], float],
) -> DimensionSelection:
    """Choose the dimension whose validation-fold energy score is smallest."""

    scores = {int(dimension): float(evaluate(int(dimension))) for dimension in candidates}
    best = min(scores, key=lambda key: scores[key])
    return DimensionSelection(dimension=int(best), scores=scores)


@dataclass(frozen=True, slots=True)
class FitReport:
    parameters: np.ndarray
    loss: float
    history: tuple[float, ...]
    iterations: int

    def as_mapping(self) -> dict[str, float | str]:
        return {
            "loss": float(self.loss),
            "iterations": float(self.iterations),
            "initial_loss": float(self.history[0]) if self.history else float("nan"),
            "parameters": ",".join(f"{value:.6f}" for value in self.parameters),
        }


def fit_by_energy_score(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    max_iter: int = 2,
) -> FitReport:
    """Minimise the energy score over a parameter vector.

    The score is the manuscript's own selection criterion; the optimiser is an
    L-BFGS-B run capped at `max_iter` so a smoke run stays a fixed, small cost.
    """

    from scipy.optimize import minimize

    start = np.asarray(initial, dtype=np.float64).reshape(-1)
    history: list[float] = [float(objective(start))]

    def callback(parameters: np.ndarray) -> None:
        history.append(float(objective(np.asarray(parameters, dtype=np.float64))))

    result = minimize(
        lambda parameters: float(objective(np.asarray(parameters, dtype=np.float64))),
        start,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": int(max_iter), "disp": False},
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    return FitReport(
        parameters=parameters,
        loss=float(result.fun),
        history=tuple(history),
        iterations=int(result.nit),
    )


def candidate_dimensions(
    low: int = LATENT_DIMENSION_RANGE[0], high: int = LATENT_DIMENSION_RANGE[1]
) -> tuple[int, ...]:
    return tuple(range(int(low), int(high) + 1))


def normal_quantile(level: float) -> float:
    return float(norm.ppf(float(level)))
