import pytest
import numpy as np
import rasterio

from insolation_model.geometry.shading import (
    _double_resolution_of_raster,
    _halve_resolution_of_raster,
)
from insolation_model.raster import Raster
from tests.conftest import make_dem_with_gradients


def _test_raster_resolution_doubler(raster: Raster):
    doubled_raster = _double_resolution_of_raster(raster)
    assert doubled_raster.arr.shape == (
        2 * raster.arr.shape[0],
        2 * raster.arr.shape[1],
    )
    assert doubled_raster.dx == raster.dx / 2
    assert doubled_raster.dy == raster.dy / 2
    for i, j in zip(range(2), range(2)):
        assert (
            doubled_raster.arr[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] == raster.arr[i, j]
        ).all()


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
def test_double_resolution_of_very_simple_raster(dx, dy):
    raster = Raster(
        arr=np.array([[1, 2], [3, 4]]),
        transform=rasterio.Affine(dx, 0, 0, 0, -dy, 0),
        crs=rasterio.crs.CRS.from_epsg(4326),
    )
    _test_raster_resolution_doubler(raster)


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 5])
@pytest.mark.parametrize("grad_x", [0, -1, 3])
@pytest.mark.parametrize("grad_y", [0, 1, -3])
@pytest.mark.parametrize("n_rows", [3, 8])
@pytest.mark.parametrize("n_cols", [5, 6])
def test_double_resolution_of_simple_raster_with_gradients(
    dx, dy, grad_x, grad_y, n_rows, n_cols
):
    raster = make_dem_with_gradients(grad_x, grad_y, dx, dy, n_rows, n_cols)
    _test_raster_resolution_doubler(raster)


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 5])
@pytest.mark.parametrize("grad_x", [0, -1])
@pytest.mark.parametrize("grad_y", [-3])
@pytest.mark.parametrize("n_rows", [12])
@pytest.mark.parametrize("n_cols", [12])
def test_raster_resolution_changes_round_trip(dx, dy, grad_x, grad_y, n_rows, n_cols):
    raster = make_dem_with_gradients(grad_x, grad_y, dx, dy, n_rows, n_cols)
    doubled_raster = _double_resolution_of_raster(raster)
    halved_raster = _halve_resolution_of_raster(doubled_raster)
    np.testing.assert_array_almost_equal(halved_raster.arr, raster.arr)
