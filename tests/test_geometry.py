"""Template frame, geometry summaries and the segmentation perturbations."""

from __future__ import annotations

import numpy as np

from bmstf.geometry.perturbations import (
    PerturbationSpec,
    boundary_condition_scale,
    mask_perturbations,
    smooth_radial_field,
    topology_class,
    topology_flag,
)
from bmstf.geometry.shapes import (
    ShapeSpec,
    geometry_from_state,
    solve_ilt_thickness,
    target_ilt_fraction,
)
from bmstf.geometry.summaries import (
    GEOMETRY_FEATURES,
    covariate_vector,
    maximum_diameter_mm,
    summarise,
    thrombus_fraction,
)
from bmstf.geometry.template import annular_volumes, canonicalise, resample_radii


def test_template_frame_is_orthonormal(template) -> None:
    assert template.nodes == 48
    assert abs(template.spacing_mm - 110.0 / 47.0) < 1.0e-12
    assert np.allclose(np.linalg.norm(template.tangents, axis=1), 1.0)
    assert np.allclose(template.tangents[:, 0], 1.0)


def test_resample_interpolates_in_arc_length(template) -> None:
    centers = np.asarray([0.0, 55.0, 110.0])
    radii = np.asarray([10.0, 30.0, 10.0])
    mapped = resample_radii(centers, radii, template)
    assert mapped.shape == (template.nodes,)
    assert abs(mapped[0] - 10.0) < 1.0e-12
    assert abs(mapped[-1] - 10.0) < 1.0e-12
    reference = np.interp(template.centers[:, 0], centers, radii)
    assert np.max(np.abs(mapped - reference)) < 1.0e-12
    step = (30.0 - 10.0) * template.spacing_mm / 55.0
    assert abs(float(np.max(mapped)) - 30.0) < step


def test_canonicalise_orders_the_three_layers(template) -> None:
    centers = np.linspace(0.0, 110.0, 20)
    geometry = canonicalise(
        "case", centers, np.full(20, 12.0), np.full(20, 14.0), np.full(20, 16.0), template
    )
    assert np.all(geometry.lumen_radius <= geometry.thrombus_radius + 1.0e-12)
    assert np.all(geometry.thrombus_radius <= geometry.wall_radius + 1.0e-12)


def test_geometry_summaries_are_consistent(geometry) -> None:
    summary = summarise(geometry)
    assert abs(summary.diameter_mm - maximum_diameter_mm(geometry)) < 1.0e-12
    assert 0.0 <= summary.ilt_fraction <= 1.0
    assert summary.sac_volume_cm3 > 0.0
    assert abs(summary.tortuosity - 1.0) < 1.0e-9
    assert summary.as_vector().shape == (len(GEOMETRY_FEATURES),)


def test_covariate_vector_carries_the_site(geometry) -> None:
    summary = summarise(geometry)
    first = covariate_vector(summary, 0)
    second = covariate_vector(summary, 2)
    assert first.shape == second.shape == (7,)
    assert float(np.sum(first[4:])) == 1.0
    assert abs(float(first[4] - second[4]) - 1.0) < 1.0e-12


def test_annular_volumes_increase_with_the_layer(geometry) -> None:
    lumen, thrombus, sheath = annular_volumes(geometry)
    assert lumen > 0.0 and thrombus > 0.0 and sheath > lumen


def test_ilt_solver_hits_its_target() -> None:
    for burden in (0.0, 0.2, 0.5, 0.8, 0.98):
        geometry = geometry_from_state(f"ilt-{burden}", 46.0, burden)
        target = target_ilt_fraction(burden, ShapeSpec())
        assert abs(thrombus_fraction(geometry) - target) < 5.0e-3


def test_ilt_thickness_is_monotone() -> None:
    lumen = np.full(48, 20.0)
    low = solve_ilt_thickness(lumen, 0.1, ShapeSpec())
    high = solve_ilt_thickness(lumen, 0.4, ShapeSpec())
    assert low < high


def test_smooth_radial_field_is_zero_mean_and_correlated() -> None:
    field = smooth_radial_field(48, 2.34, np.random.default_rng(1), 0.12, 9.0)
    assert field.shape == (48,)
    assert abs(float(np.mean(field))) < 1.0e-12
    assert float(np.std(field)) > 0.0
    neighbour = float(np.corrcoef(field[:-1], field[1:])[0, 1])
    assert neighbour > 0.5


def test_perturbations_move_the_surface_but_keep_it_ordered(geometry) -> None:
    perturbed = mask_perturbations(geometry, PerturbationSpec(3), np.random.default_rng(2))
    assert len(perturbed) == 3
    for member in perturbed:
        assert np.all(member.lumen_radius <= member.thrombus_radius + 1.0e-12)
        assert np.all(member.thrombus_radius <= member.wall_radius + 1.0e-12)
        assert float(np.max(np.abs(member.wall_radius - geometry.wall_radius))) > 0.0


def test_topology_flag_and_class_are_bounded(geometry) -> None:
    assert topology_flag(geometry) in (0, 1)
    assert 0 <= topology_class(geometry) < 4
    assert 0.0 < boundary_condition_scale(geometry) <= 2.0
