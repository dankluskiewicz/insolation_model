import pytest
import numpy as np
from insolation_model.geometry.topography import (
    dem_to_gradient,
    dem_to_surface_normal_unit_direction,
)
from tests.conftest import make_dem_with_gradients


@pytest.mark.parametrize("prescribed_grad_x", [0, 1, -1, 3])
@pytest.mark.parametrize("prescribed_grad_y", [0, 1, -1, 3])
@pytest.mark.parametrize("dx", [1, 2, 4])
def test_dem_to_gradient(prescribed_grad_x, prescribed_grad_y, dx):
    dem = make_dem_with_gradients(prescribed_grad_x, prescribed_grad_y, dx, dx)
    measured_grad_x, measured_grad_y = dem_to_gradient(dem)
    assert np.isclose(measured_grad_x, prescribed_grad_x).all()
    assert np.isclose(measured_grad_y, prescribed_grad_y).all()


def _gradients_to_grad_vector(grad_x, grad_y):
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.array([grad_x, grad_y, gradient_magnitude * np.hypot(grad_x, grad_y)])


@pytest.mark.parametrize("prescribed_grad_x", [0, 1, -1, 3])
@pytest.mark.parametrize("prescribed_grad_y", [0, 1, -1, 3])
@pytest.mark.parametrize("dx", [1, 2])
def test_dem_to_surface_normal_unit_direction(prescribed_grad_x, prescribed_grad_y, dx):
    dem = make_dem_with_gradients(prescribed_grad_x, prescribed_grad_y, dx, dx)
    unit_norm = dem_to_surface_normal_unit_direction(dem)
    # need an array of vectors that point along the surface gradient
    # I'm using caps to denote the gradient array that I infer form the DEM, as opposed to the prescribed gradient values.
    GradX, GradY = dem_to_gradient(dem)
    grad_vector = _gradients_to_grad_vector(GradX, GradY)
    assert np.isclose(np.linalg.norm(unit_norm, axis=0), 1.0, atol=1e-5).all(), (
        "The surface normal unit direction is not a unit vector. \n"
        f"{prescribed_grad_x=}, {prescribed_grad_y=}\n"
        f"unit_norm: {unit_norm}\n"
    )
    assert np.isclose(np.sum(unit_norm * grad_vector, axis=0), 0.0, atol=1e-5).all(), (
        "The surface normal unit direction is not perpendicular to the gradient",
        # f"grad_vector: {grad_vector}\n"
        # f"unit_norm: {unit_norm}\n"
        # f"gradient_magnitude: {gradient_magnitude}\n"
    )
    assert (unit_norm[2] > 0.0).all(), (
        "The surface normal unit direction is not pointing upwards."
    )
    assert (np.sign(unit_norm[0]) == -np.sign(prescribed_grad_x)).all(), (
        "The surface normal unit direction is not pointing in the correct x direction."
    )
    assert (np.sign(unit_norm[1]) == -np.sign(prescribed_grad_y)).all(), (
        "The surface normal unit direction is not pointing in the correct y direction."
    )


def test_dem_to_surface_normal_unit_direction_does_not_integer_truncate_when_first_pixel_flat():
    """Regression test: when the first pixel has zero gradient, we must not infer int outputs and truncate."""
    dem = make_dem_with_gradients(
        grad_x=0.5, grad_y=-0.25, dx=1.0, dy=1.0, n_rows=6, n_cols=7
    )
    dem.arr[0, 0] = 0.0
    dem.arr[0, 1] = 0.0
    dem.arr[1, 0] = 0.0  # make a small flat corner

    unit_norm = dem_to_surface_normal_unit_direction(dem)
    assert unit_norm.dtype.kind == "f"
    # Away from the forced-flat corner, x/y components should be non-zero for non-zero gradients.
    assert np.any(np.abs(unit_norm[0, 2:, 2:]) > 0)
    assert np.any(np.abs(unit_norm[1, 2:, 2:]) > 0)
