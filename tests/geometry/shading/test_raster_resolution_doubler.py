import pytest
import numpy as np
import rasterio

from insolation_model.geometry.shading import (
    _double_resolution_of_raster,
    _double_resolution_of_array,
)
from insolation_model.geometry.topography import dem_to_gradient
from insolation_model.raster import Raster
from tests.conftest import make_dem_with_gradients


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
def test_double_resolution_of_very_simple_raster(dx, dy):
    raster = Raster(
        arr=np.array([[1, 2], [3, 4]]),
        transform=rasterio.Affine(dx, 0, 0, 0, -dy, 0),
        crs=rasterio.crs.CRS.from_epsg(4326),
    )
    doubled_raster = _double_resolution_of_raster(raster)
    assert doubled_raster.arr.shape == (4, 4)
    assert doubled_raster.arr[0, 0] == 1
    assert doubled_raster.arr[0, 1] == 1.5


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 5])
@pytest.mark.parametrize("grad_x", [0, -1, 3])
@pytest.mark.parametrize("grad_y", [0, 1, -3])
@pytest.mark.parametrize("n_rows", [3, 8])
@pytest.mark.parametrize("n_cols", [2, 5])
def test_double_resolution_of_simple_raster_with_gradients(
    dx, dy, grad_x, grad_y, n_rows, n_cols
):
    raster = make_dem_with_gradients(grad_x, grad_y, dx, dy, n_rows, n_cols)
    doubled_raster = _double_resolution_of_raster(raster)

    assert doubled_raster.arr.shape == (2 * n_rows, 2 * n_cols)
    assert doubled_raster.dx == dx / 2
    assert doubled_raster.dy == dy / 2
    doubled_grad_x, doubled_grad_y = dem_to_gradient(doubled_raster)
    assert all(doubled_grad_x == _double_resolution_of_array(grad_x)).all()
    assert all(doubled_grad_y == _double_resolution_of_array(grad_y)).all()
