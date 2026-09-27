"""Boundary-condition-conditioned physics-attention shear surrogate.

Ref: Sec. "Here we make the identification explicit" (a boundary-condition-conditioned
physics-attention surrogate, marginalised over the unmeasured boundary conditions);
Algorithm 1 line 1 (f_theta(g, b) in R^p). The surrogate is a parity component of
the design and is never the mechanism under test.

The map is written in closed form so that the low-shear burden it returns can be
recomputed by enumeration for any geometry and inflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bmstf.geometry.template import ScanGeometry

SHEAR_CHANNELS: tuple[str, ...] = (
    "diameter_mm",
    "low_shear_area_mm2",
    "burden",
    "osi_mean",
)

LOW_SHEAR_THRESHOLD_PA = 0.4


@dataclass(frozen=True, slots=True)
class SurrogateParameters:
    """The free parameters of the surrogate, held as plain arrays."""

    shear_gain: float = 1.0
    threshold_pa: float = LOW_SHEAR_THRESHOLD_PA
    oscillation_gain: float = 0.55
    waveform_gain: float = 0.30
    attention_temperature: float = 3.0

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "shear_gain": np.asarray([self.shear_gain]),
            "threshold_pa": np.asarray([self.threshold_pa]),
            "oscillation_gain": np.asarray([self.oscillation_gain]),
            "waveform_gain": np.asarray([self.waveform_gain]),
            "attention_temperature": np.asarray([self.attention_temperature]),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> SurrogateParameters:
        def scalar(key: str, fallback: float) -> float:
            if key not in arrays:
                return fallback
            return float(np.asarray(arrays[key]).reshape(-1)[0])

        return cls(
            shear_gain=scalar("shear_gain", 1.0),
            threshold_pa=scalar("threshold_pa", LOW_SHEAR_THRESHOLD_PA),
            oscillation_gain=scalar("oscillation_gain", 0.55),
            waveform_gain=scalar("waveform_gain", 0.30),
            attention_temperature=scalar("attention_temperature", 3.0),
        )

    def perturbed(self, factor: float) -> SurrogateParameters:
        return SurrogateParameters(
            shear_gain=self.shear_gain * factor,
            threshold_pa=self.threshold_pa,
            oscillation_gain=self.oscillation_gain,
            waveform_gain=self.waveform_gain,
            attention_temperature=self.attention_temperature,
        )


@dataclass(frozen=True, slots=True)
class NodeShear:
    """Per-node wall shear before the summaries are reduced."""

    tawss: np.ndarray
    oscillatory_index: np.ndarray
    area_mm2: np.ndarray
    attention: np.ndarray


@dataclass(frozen=True, slots=True)
class SurrogateOutput:
    values: np.ndarray
    nodes: NodeShear

    def as_mapping(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(SHEAR_CHANNELS, self.values, strict=True)}


@dataclass(frozen=True, slots=True)
class PhysicsAttentionSurrogate:
    """A single-head attention read-out over the template nodes, tied to Poiseuille shear."""

    parameters: SurrogateParameters = field(default_factory=SurrogateParameters)

    def node_shear(self, geometry: ScanGeometry, boundary: np.ndarray) -> NodeShear:
        amplitude = float(np.asarray(boundary).reshape(-1)[0])
        period = float(np.asarray(boundary).reshape(-1)[1])
        resistance = float(np.asarray(boundary).reshape(-1)[2])
        viscosity = float(np.asarray(boundary).reshape(-1)[3])
        steepness = float(np.asarray(boundary).reshape(-1)[4])

        radius_mm = np.maximum(geometry.lumen_radius, 0.5)
        wall = np.maximum(geometry.wall_radius, radius_mm)
        step = geometry.template.spacing_mm
        area = 2.0 * np.pi * wall * step

        poiseuille = 4.0 * viscosity * amplitude / (np.pi * radius_mm)
        waveform = 1.0 + self.parameters.waveform_gain * np.tanh(steepness * (1.0 - 60.0 * period))
        outlet = 1.0 - 0.25 * (1.0 - resistance)
        taper = 1.0 - 0.35 * np.tanh((wall - radius_mm) / np.maximum(radius_mm, 1.0e-3))
        tawss = self.parameters.shear_gain * poiseuille * waveform * outlet * taper
        tawss = np.maximum(tawss, 1.0e-6)

        spread = np.std(tawss)
        local = (tawss - float(np.mean(tawss))) / max(spread, 1.0e-9)
        oscillatory = 1.0 / (1.0 + np.exp(-self.parameters.oscillation_gain * local))
        oscillatory = np.clip(oscillatory, 0.0, 1.0)

        weights = np.exp(self.parameters.attention_temperature * (-local))
        attention = weights / float(np.sum(weights))
        return NodeShear(
            tawss=tawss, oscillatory_index=oscillatory, area_mm2=area, attention=attention
        )

    def __call__(self, geometry: ScanGeometry, boundary: np.ndarray) -> np.ndarray:
        return self.node_shear(geometry, boundary).tawss

    def summarise(
        self, nodes: NodeShear, diameter_mm: float, mask: np.ndarray | None = None
    ) -> np.ndarray:
        active = (
            np.ones(nodes.tawss.shape[0], dtype=bool) if mask is None else np.asarray(mask, bool)
        )
        low = active & (nodes.tawss < self.parameters.threshold_pa)
        total_area = float(np.sum(nodes.area_mm2[active]))
        low_area = float(np.sum(nodes.area_mm2[low]))
        burden = low_area / total_area if total_area > 0.0 else 0.0
        osi_mean = (
            float(np.sum(nodes.oscillatory_index[low] * nodes.area_mm2[low]) / low_area)
            if low_area > 0.0
            else 0.0
        )
        return np.asarray([float(diameter_mm), low_area, burden, osi_mean], dtype=np.float64)

    def evaluate(self, geometry: ScanGeometry, boundary: np.ndarray) -> SurrogateOutput:
        nodes = self.node_shear(geometry, boundary)
        diameter = 2.0 * float(np.max(geometry.lumen_radius[geometry.node_mask]))
        return SurrogateOutput(
            values=self.summarise(nodes, diameter, geometry.node_mask), nodes=nodes
        )

    def enumerate_burden(self, geometry: ScanGeometry, boundary: np.ndarray) -> float:
        """Independent recomputation of the burden by direct enumeration."""

        nodes = self.node_shear(geometry, boundary)
        active = geometry.node_mask
        low = np.asarray(nodes.tawss < self.parameters.threshold_pa) & active
        total = float(np.sum(nodes.area_mm2[active]))
        if total <= 0.0:
            return 0.0
        return float(np.sum(nodes.area_mm2[low]) / total)
