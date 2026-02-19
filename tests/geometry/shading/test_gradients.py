import pytest
import numpy as np

from insolation_model.geometry.shading import _add_gradient_to_dem
from insolation_model.geometry.topography import dem_to_gradient
from tests.conftest import make_flat_dem


@pytest.mark.parametrize("grad_x", [0, 1, -1, 3])
@pytest.mark.parametrize("grad_y", [0, 1, -1, 3])
def test_add_gradient_to_dem(grad_x, grad_y):
    dem = make_flat_dem()
    dem_with_gradient = _add_gradient_to_dem(dem, grad_x, grad_y)
    measured_grad_x, measured_grad_y = dem_to_gradient(dem_with_gradient)
    assert np.isclose(measured_grad_x, grad_x).all()
    assert np.isclose(measured_grad_y, grad_y).all()
