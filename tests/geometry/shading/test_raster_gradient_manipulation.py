import pytest
import numpy as np

from insolation_model.geometry.shading import (
    _add_y_gradient,
    _gradient_from_elevation_angle,
)
from insolation_model.geometry.topography import dem_to_gradient
from tests.conftest import make_dem_with_gradients, make_flat_dem


def _rad(degrees: float) -> float:
    return degrees * np.pi / 180


@pytest.mark.parametrize("gradient", [0, 1, -1, 3])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
def test_add_y_gradient(gradient, dx, dy):
    flat_dem = make_flat_dem(dx, dy)
    expected_dem = make_dem_with_gradients(0, gradient, dx, dy)
    new_dem = _add_y_gradient(flat_dem, gradient)
    print(new_dem.arr)
    np.testing.assert_array_almost_equal(new_dem.arr, expected_dem.arr)
    measured_grad_x, measured_grad_y = dem_to_gradient(new_dem)
    assert np.isclose(measured_grad_y, gradient).all()
    assert np.isclose(measured_grad_x, 0).all()


@pytest.mark.parametrize("elevation_angle", [0, 15, 37, 89])
def test_gradient_from_elevation_angle(elevation_angle):
    flat_dem = make_flat_dem()
    gradient = _gradient_from_elevation_angle(elevation_angle)
    new_dem = _add_y_gradient(flat_dem, gradient)
    _, ymin, _, ymax = flat_dem.bounds
    Dy = ymax - ymin - flat_dem.dy
    Dz = new_dem.arr.max() - new_dem.arr.min()
    measured_gradient = Dz / Dy
    expected_gradient = np.tan(_rad(elevation_angle))
    assert np.isclose(measured_gradient, expected_gradient), (
        f"Measured gradient {measured_gradient} does not match expected gradient {expected_gradient}"
    )
