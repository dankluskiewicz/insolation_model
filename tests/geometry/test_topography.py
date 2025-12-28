import pytest
import numpy as np
import rasterio
from insolation_model.raster import Raster
from insolation_model.geometry.topography import dem_to_gradient

dx, dy = 1, -2
N, M = 10, 10
transform = rasterio.Affine(dx, 0, 0, 0, dy, 0)


dem_with_x_gradient = Raster(
    arr=np.vstack([np.arange(M)] * N),
    transform=transform,
    crs=rasterio.crs.CRS.from_epsg(4326),
)

dem_with_y_gradient = Raster(
    arr=dem_with_x_gradient.arr.transpose(),
    transform=transform,
    crs=rasterio.crs.CRS.from_epsg(4326),
)

dem_with_x_and_y_gradients = Raster(
    arr=dem_with_x_gradient.arr + dem_with_y_gradient.arr,
    transform=transform,
    crs=rasterio.crs.CRS.from_epsg(4326),
)


@pytest.mark.parametrize("dem, expected_dx, expected_dy", [
    (dem_with_x_gradient, 1 / dx, 0),
    (dem_with_y_gradient, 0, 1 / dy),
    (dem_with_x_and_y_gradients, 1 / dx, 1 / dy),
])
def test_dem_to_gradient(dem, expected_dx, expected_dy):
    grad_x, grad_y = dem_to_gradient(dem)
    assert np.isclose(grad_x, expected_dx).all()
    assert np.isclose(grad_y, expected_dy).all()
