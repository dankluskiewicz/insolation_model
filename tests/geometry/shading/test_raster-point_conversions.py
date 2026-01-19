import functools
import pytest

import numpy as np

from insolation_model.geometry.shading import (
    _point_representation_of_dem,
    _unflatten_vector_to_raster_dimensions,
    _rotate_points_around_z_axis,
    _raster_representation_of_points_mean_z,
)
from tests.conftest import make_dem_with_gradients, make_flat_dem, make_dem_with_step


def test_points_represent_cell_centers_and_values():
    """Test that points represent cell centers (not corners)."""
    n_rows, n_cols = 2, 2
    dx, dy = 2.0, 3.0
    origin_x, origin_y = 0.0, 10.0
    dem = make_dem_with_gradients(
        grad_x=0,
        grad_y=0,
        dx=dx,
        dy=dy,
        n_rows=n_rows,
        n_cols=n_cols,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    X, Y, Z = _point_representation_of_dem(dem)

    expected_x = np.array([origin_x + dx * 0.5, origin_x + dx * 1.5])
    expected_y = np.array([origin_y - dy * 0.5, origin_y - dy * 1.5])
    expected_X = np.vstack([expected_x] * n_rows).flatten()
    expected_Y = np.vstack([expected_y] * n_rows).transpose().flatten()
    expected_Z = dem.arr.flatten()

    np.testing.assert_array_almost_equal(
        X, expected_X, err_msg="Points should represent cell centers"
    )
    np.testing.assert_array_almost_equal(
        Y, expected_Y, err_msg="Points should represent cell centers"
    )
    np.testing.assert_array_almost_equal(
        Z, expected_Z, err_msg="Points should represent cell values"
    )


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
@pytest.mark.parametrize("n_rows", [3, 8])
@pytest.mark.parametrize("n_cols", [3, 4, 9])
@pytest.mark.parametrize("origin_x", [0, 10])
@pytest.mark.parametrize("origin_y", [0, -10])
@pytest.mark.parametrize(
    "dem_factory",
    [
        make_flat_dem,
        functools.partial(make_dem_with_gradients, grad_x=1, grad_y=2),
        functools.partial(make_dem_with_step, step_size=1, start_index=1, stop_index=3),
        functools.partial(
            make_dem_with_step, step_size=-1, start_index=1, stop_index=3, step_axis=1
        ),
    ],
)
def test_raster_point_round_trip(
    dx, dy, n_rows, n_cols, origin_x, origin_y, dem_factory
):
    # make sure that I can get back the values from a raster after converting to points and back
    dem = dem_factory(
        dx=dx,
        dy=dy,
        n_rows=n_rows,
        n_cols=n_cols,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    X, Y, Z = _point_representation_of_dem(dem)
    unflattened_Z = _unflatten_vector_to_raster_dimensions(Z, n_rows, n_cols)
    np.testing.assert_array_almost_equal(unflattened_Z, dem.arr)

    dem_after_round_trip = _raster_representation_of_points_mean_z(
        np.stack([X, Y, Z], axis=0), dx, dy, *dem.bounds, crs=dem.crs
    )
    np.testing.assert_array_almost_equal(dem_after_round_trip.arr, dem.arr)


@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 2])
@pytest.mark.parametrize("n_rows", [3, 8])
@pytest.mark.parametrize("n_cols", [3, 4, 9])
@pytest.mark.parametrize("origin_x", [0, 10])
@pytest.mark.parametrize("origin_y", [0, -10])
@pytest.mark.parametrize(
    "dem_factory",
    [
        make_flat_dem,
        functools.partial(make_dem_with_gradients, grad_x=1, grad_y=2),
        functools.partial(make_dem_with_step, step_size=1, start_index=1, stop_index=3),
        functools.partial(
            make_dem_with_step, step_size=-1, start_index=1, stop_index=3, step_axis=1
        ),
    ],
)
@pytest.mark.parametrize("rotation_angle", [0, 15, 37, 90, 180, 211])
def test_raster_point_round_trip_with_rotation(
    dx, dy, n_rows, n_cols, origin_x, origin_y, dem_factory, rotation_angle
):
    dem = dem_factory(
        dx=dx,
        dy=dy,
        n_rows=n_rows,
        n_cols=n_cols,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    X, Y, Z = _point_representation_of_dem(dem)
    rotated_XYZ = _rotate_points_around_z_axis(
        np.stack([X, Y, Z], axis=0), rotation_angle
    )
    unrotated_X, unrotated_Y, unrotated_Z = _rotate_points_around_z_axis(
        rotated_XYZ, -rotation_angle
    )
    np.testing.assert_array_almost_equal(unrotated_X, X)
    np.testing.assert_array_almost_equal(unrotated_Y, Y)
    np.testing.assert_array_almost_equal(unrotated_Z, Z)
    unflattened_Z = _unflatten_vector_to_raster_dimensions(unrotated_Z, n_rows, n_cols)
    np.testing.assert_array_almost_equal(unflattened_Z, dem.arr)
