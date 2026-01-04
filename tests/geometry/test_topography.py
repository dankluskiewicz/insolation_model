import pytest
import numpy as np
from insolation_model.geometry.topography import (
    dem_to_gradient,
    dem_to_surface_normal_unit_direction,
    _gradient_vector_to_surface_normal_unit_direction,
)
from tests.conftest import make_dem_with_gradients


@pytest.mark.parametrize("prescribed_grad_x", [0, 1, -1, 3])
@pytest.mark.parametrize("prescribed_grad_y", [0, 1, -1, 3])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 4])
def test_dem_to_gradient(prescribed_grad_x, prescribed_grad_y, dx, dy):
    dem = make_dem_with_gradients(prescribed_grad_x, prescribed_grad_y, dx, dy)
    measured_grad_x, measured_grad_y = dem_to_gradient(dem)
    assert np.isclose(measured_grad_x, prescribed_grad_x).all()
    assert np.isclose(measured_grad_y, prescribed_grad_y).all()


@pytest.mark.parametrize("prescribed_grad_x", [0, 1, -1, 3])
@pytest.mark.parametrize("prescribed_grad_y", [0, 1, -1, 3])
def test_gradient_vector_to_surface_normal_unit_direction(
    prescribed_grad_x, prescribed_grad_y
):
    unit_norm = np.array(
        _gradient_vector_to_surface_normal_unit_direction(
            prescribed_grad_x, prescribed_grad_y
        )
    )
    # need a vector that points along the surface gradient
    gradient_magnitude = np.sqrt(prescribed_grad_x**2 + prescribed_grad_y**2)
    grad_vector = np.array(
        [
            prescribed_grad_x,
            prescribed_grad_y,
            -gradient_magnitude * np.hypot(prescribed_grad_x, prescribed_grad_y),
        ]
    )
    assert np.isclose(np.linalg.norm(unit_norm), 1.0, atol=1e-5).all(), (
        "The surface normal unit direction is not a unit vector"
    )
    assert np.isclose(np.dot(unit_norm, grad_vector), 0.0, atol=1e-5).all(), (
        "The surface normal unit direction is not perpendicular to the gradient. \n"
        f"{grad_vector=}\n"
        f"unit_norm: {unit_norm}\n"
    )


@pytest.mark.parametrize("prescribed_grad_x", [0, 1, -1, 3])
@pytest.mark.parametrize("prescribed_grad_y", [0, 1, -1, 3])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 4])
def test_dem_to_surface_normal_unit_direction(
    prescribed_grad_x, prescribed_grad_y, dx, dy
):
    dem = make_dem_with_gradients(prescribed_grad_x, prescribed_grad_y, dx, dy)
    unit_norm = dem_to_surface_normal_unit_direction(dem)
    # need an array of vectors that point along the surface gradient
    # I'm using caps to denote the gradient array that I infer form the DEM, as opposed to the prescribed gradient values.
    GradX, GradY = dem_to_gradient(dem)
    gradient_magnitude = np.sqrt(GradX**2 + GradY**2)
    grad_vector = np.array([GradX, GradY, -gradient_magnitude * np.hypot(GradX, GradY)])
    assert np.isclose(np.linalg.norm(unit_norm, axis=0), 1.0, atol=1e-5).all(), (
        "The surface normal unit direction is not a unit vector. \n"
        f"{prescribed_grad_x=}, {prescribed_grad_y=}\n"
        f"unit_norm: {unit_norm}\n"
    )
    assert np.isclose(np.sum(unit_norm * grad_vector, axis=0), 0.0, atol=1e-5).all(), (
        "The surface normal unit direction is not perpendicular to the gradient"
    )
