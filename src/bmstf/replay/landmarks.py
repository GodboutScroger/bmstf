"""Landmark construction for the retrospective policy replay.

Ref: Algorithm 3 line 1 and line 4: the landmark index k_L = 2, the per-visit
window W_ik = (t_ik, t_ik + tau*_ik], and the restriction to CTA visits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LANDMARK_INDEX = 2


@dataclass(frozen=True, slots=True)
class Landmark:
    index: int
    month: float
    channel: str

    def as_mapping(self) -> dict[str, float | str]:
        return {"index": float(self.index), "month": float(self.month), "channel": self.channel}


def landmark_indices(channels: tuple[str, ...], start: int = LANDMARK_INDEX) -> tuple[int, ...]:
    """Every CTA visit from the landmark onwards carries a replay decision."""

    return tuple(index for index in range(int(start), len(channels)) if channels[index] == "CTA")


def window(start: float, horizon: float) -> tuple[float, float]:
    return (float(start), float(start) + float(horizon))


def inside_window(month: float | None, start: float, horizon: float) -> bool:
    if month is None:
        return False
    return bool(float(start) < float(month) <= float(start) + float(horizon))


def has_minimum_history(observations: int, minimum: int = LANDMARK_INDEX) -> bool:
    return bool(int(observations) >= int(minimum))


def landmark_summary(months: np.ndarray, channels: tuple[str, ...]) -> list[Landmark]:
    grid = np.asarray(months, dtype=np.float64)
    return [
        Landmark(index=index, month=float(grid[index]), channel=str(channels[index]))
        for index in landmark_indices(channels)
    ]
