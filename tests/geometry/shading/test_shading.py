import pytest
import numpy as np

from insolation_model.raster import Raster
from insolation_model.geometry.topography import dem_to_gradient
from insolation_model.geometry.shading import (
    _shading_mask_from_sun_at_north_horizon,
    get_shading_mask,
    _rad,
)
from tests.conftest import make_dem_with_gradients, make_dem_with_step, make_flat_dem


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
@pytest.mark.parametrize("grad_x", [1, -2, 999999])
@pytest.mark.parametrize(
    ["grad_y", "expected_mask"],
    [
        (0, False),
        (1, True),
        (-1, False),
        (3, True),
        (999999, True),
    ],
)
def test_shading_mask_from_sun_at_north_horizon(dx, dy, grad_x, grad_y, expected_mask):
    dem = make_dem_with_gradients(grad_x, grad_y, dx, dy)
    mask = _shading_mask_from_sun_at_north_horizon(dem)
    assert mask.arr.all() == expected_mask


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
@pytest.mark.parametrize("step_size", [1, -2, -99, 99])
@pytest.mark.parametrize("start_index", [0, 1, 3])
@pytest.mark.parametrize("stop_index", [4, 8])
def test_shading_mask_from_sun_at_north_horizon_with_step(
    dx, dy, step_size, start_index, stop_index
):
    dem = make_dem_with_step(
        step_size, start_index, stop_index, 0, dx, dy, n_rows=8, n_cols=4
    )
    mask = _shading_mask_from_sun_at_north_horizon(dem)
    expected_mask = np.zeros(dem.arr.shape, dtype=bool)
    expected_mask[start_index:, :] = step_size < 0
    print(dem.arr)
    print(dem_to_gradient(dem)[1])
    expected_mask[dem_to_gradient(dem)[1] > 0] = True
    print(f"{mask=}, \n{expected_mask=}")
    np.testing.assert_array_equal(mask.arr, expected_mask)


@pytest.mark.parametrize("azimuth_angle", [0, 30, 180, 275])
@pytest.mark.parametrize("elevation_angle", [0, 30, 90])
def test_flat_slope_never_shaded(azimuth_angle, elevation_angle):
    dem = make_flat_dem(1, 1)
    mask = get_shading_mask(
        dem, solar_azimuth_angle=azimuth_angle, solar_elevation_angle=elevation_angle
    )
    np.testing.assert_array_equal(mask, np.zeros(dem.arr.shape, dtype=int))


@pytest.mark.parametrize(("grad_x", "grad_y"), [(0, 1), (1, 4)])
@pytest.mark.parametrize("azimuth_angle", [0, 30, 180, 275])
def test_dem_is_never_masked_when_solar_elevation_angle_is_90(
    grad_x, grad_y, azimuth_angle
):
    dem = make_dem_with_gradients(
        grad_x, grad_y, 1, 1, n_rows=4, n_cols=5, origin_x=0, origin_y=0
    )
    np.testing.assert_array_equal(
        get_shading_mask(
            dem, solar_azimuth_angle=azimuth_angle, solar_elevation_angle=90
        ),
        np.zeros(dem.arr.shape, dtype=int),
    )


@pytest.mark.parametrize("elevation_angle", [3, 15, 37, 87])
@pytest.mark.parametrize("azimuth_angle", [0, 90, 180, 270, 1, 45, 15, 75, 265])
def test_get_shading_mask(elevation_angle, azimuth_angle):
    eps = 1
    should_be_shaded = _dem_with_slope_that_parallels_solar_elevation(
        elevation_angle + eps, azimuth_angle, 1, 1
    )
    should_not_be_shaded = _dem_with_slope_that_parallels_solar_elevation(
        elevation_angle - eps, azimuth_angle, 1, 1
    )
    np.testing.assert_array_equal(
        get_shading_mask(
            should_be_shaded,
            solar_azimuth_angle=azimuth_angle,
            solar_elevation_angle=elevation_angle,
        ),
        np.ones(should_be_shaded.arr.shape, dtype=int),
    )
    np.testing.assert_array_equal(
        get_shading_mask(
            should_not_be_shaded,
            solar_azimuth_angle=azimuth_angle,
            solar_elevation_angle=elevation_angle,
        ),
        np.zeros(should_not_be_shaded.arr.shape, dtype=int),
    )


def _gradient_for_slope_that_parallels_solar_elevation(
    elevation_angle: float, azimuth_angle: float
) -> float:
    if elevation_angle <= 0:
        raise ValueError("Elevation angle must be greater than 0")
    if elevation_angle >= 90:
        raise ValueError("Elevation angle must be less than 90")
    grad_x = np.sin(_rad(azimuth_angle)) * np.tan(_rad(elevation_angle))
    grad_y = np.cos(_rad(azimuth_angle)) * np.tan(_rad(elevation_angle))
    return grad_x, grad_y


def _dem_with_slope_that_parallels_solar_elevation(
    elevation_angle: float,
    azimuth_angle: float,
    dx: float,
    dy: float,
    n_rows: int = 4,
    n_cols: int = 5,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Raster:
    grad_x, grad_y = _gradient_for_slope_that_parallels_solar_elevation(
        elevation_angle, azimuth_angle
    )
    return make_dem_with_gradients(
        grad_x, grad_y, dx, dy, n_rows, n_cols, origin_x, origin_y
    )
