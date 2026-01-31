import pytest
import numpy as np

from insolation_model.raster import Raster
from insolation_model.geometry.shading import make_dem_with_gradients
from tests.test_data.context import dem_path


@pytest.fixture
def dem():
    return Raster.from_tif(dem_path)


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
        dem.arr[start_index:stop_index, :] += np.stack(
            [step_size * np.arange(1, step_length + 1) / step_length] * n_cols,
            axis=1,
        )
        dem.arr[stop_index:, :] = step_size
    elif step_axis == 1:
        dem.arr[:, start_index:stop_index] += np.stack(
            [step_size * np.arange(1, step_length + 1) / step_length] * n_rows,
            axis=0,
        )
        dem.arr[:, stop_index:] = step_size
    return dem
