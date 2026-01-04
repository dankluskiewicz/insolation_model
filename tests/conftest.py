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


def make_flat_dem(
    dx: float = 1.0,
    dy: float = 1.0,
    n_rows: int = 4,
    n_cols: int = 5,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Raster:
    return make_dem_with_gradients(0, 0, dx, dy, n_rows, n_cols, origin_x, origin_y)


def make_dem_with_step(
    step_size: float,
    start_index: int,
    stop_index: int,
    step_axis: int = 0,
    dx: float = 1.0,
    dy: float = 1.0,
    n_rows: int = 4,
    n_cols: int = 5,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Raster:
    dem = make_flat_dem(dx, dy, n_rows, n_cols, origin_x, origin_y)
    step_length = stop_index - start_index
    if step_axis == 0:
        dem.arr[start_index:stop_index, :] += (
            step_size * np.arange(step_length) / step_length
        )
        dem.arr[stop_index:, :] = step_size
    elif step_axis == 1:
        dem.arr[:, start_index:stop_index] += (
            step_size * np.arange(step_length) / step_length
        )
        dem.arr[:, stop_index:] = step_size
    return dem
