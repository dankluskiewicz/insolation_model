import numpy as np
from insolation_model.geometry.shading import _point_representation_of_dem
from tests.conftest import make_dem_with_gradients


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
