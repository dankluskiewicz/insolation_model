import pytest
import numpy as np
import rasterio

from insolation_model.geometry.shading import _double_resolution_of_raster
from insolation_model.raster import Raster


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
def test_double_resolution_of_raster(dx, dy):
    raster = Raster(
        arr=np.array([[1, 2], [3, 4]]),
        transform=rasterio.Affine(dx, 0, 0, 0, -dy, 0),
        crs=rasterio.crs.CRS.from_epsg(4326),
    )
    doubled_raster = _double_resolution_of_raster(raster)
    assert doubled_raster.arr.shape == (4, 4)
    assert doubled_raster.arr[0, 0] == 1
    assert doubled_raster.arr[0, 1] == 1.5
