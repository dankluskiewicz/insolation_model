import pytest
import numpy as np
import rasterio

from insolation_model.raster import Raster
from tests.test_data.context import dem_path


@pytest.fixture
def dem():
    return Raster.from_tif(dem_path)


def make_dem_with_gradients(
    grad_x: float,
    grad_y: float,
    dx: float,
    dy: float,
    n_rows: int = 4,
    n_cols: int = 5,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Raster:
    """Create a test DEM with prescribed gradients."""
    transform = rasterio.Affine(dx, 0, origin_x, 0, -dy, origin_y)
    return Raster(
        arr=(
            np.vstack([np.arange(n_cols)] * n_rows) * grad_x * dx
            - np.vstack([np.arange(n_rows)] * n_cols).transpose() * grad_y * dy
        ),
        transform=transform,
        crs=rasterio.crs.CRS.from_epsg(4326),
    )
