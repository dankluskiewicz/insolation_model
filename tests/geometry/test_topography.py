import pytest
import numpy as np
import rasterio
from insolation_model.raster import Raster
from insolation_model.geometry.topography import dem_to_gradient


def make_dem_with_gradients(grad_x, grad_y, dx, dy):
    N, M = 10, 20
    transform = rasterio.Affine(dx, 0, 0, 0, dy, 0)
    return Raster(
        arr = (
            np.vstack([np.arange(M)] * N) * grad_x * dx
            + np.vstack([np.arange(N)] * M).transpose() * grad_y * dy
        ),
        transform=transform,
        crs=rasterio.crs.CRS.from_epsg(4326),
    )


@pytest.mark.parametrize("grad_x", [0, 1, -1, 3])
@pytest.mark.parametrize("grad_y", [0, 1, -1, 3])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 4])
def test_dem_to_gradient(grad_x, grad_y, dx, dy):
    dem = make_dem_with_gradients(grad_x, grad_y, dx, dy)
    measured_grad_x, measured_grad_y = dem_to_gradient(dem)
    assert np.isclose(measured_grad_x, grad_x).all()
    assert np.isclose(measured_grad_y, grad_y).all()

