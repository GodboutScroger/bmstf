"""Seeded random-number streams.

Ref: Code availability (the seeds and the run record), Algorithm 2 line 17.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Streams:
    """Independent streams for the cohort process, the sensor and the forecast."""

    generator: np.random.Generator
    sensor: np.random.Generator
    forecast: np.random.Generator
    resample: np.random.Generator

    def spawn(self, tag: str) -> np.random.Generator:
        seed = int(np.random.SeedSequence(self.seed_bits()).generate_state(1)[0])
        return np.random.default_rng([seed, _tag_offset(tag)])

    def seed_bits(self) -> int:
        return int(self.generator.integers(0, 2**63 - 1))


def _tag_offset(tag: str) -> int:
    return int(np.frombuffer(tag.encode("utf-8")[:8].ljust(8, b"\x00"), dtype="<u8")[0])


def set_seed(seed: int) -> Streams:
    """Derive four independent streams from one seed so a run is reproducible."""

    root = np.random.SeedSequence(seed)
    children = root.spawn(4)
    return Streams(
        generator=np.random.default_rng(children[0]),
        sensor=np.random.default_rng(children[1]),
        forecast=np.random.default_rng(children[2]),
        resample=np.random.default_rng(children[3]),
    )


def seed_sequence(seed: int, count: int) -> list[int]:
    return [int(value) for value in np.random.SeedSequence(seed).generate_state(count)]
