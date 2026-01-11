import numpy as np
import rasterio
import pyproj
from ..raster import Raster
from .topography import dem_to_gradient


def get_shading_mask(
    dem: Raster, solar_azimuth_angle: float, solar_elevation_angle: float
) -> np.ndarray:
    """Get the shading mask for a DEM, given a solar position.
    The solar position is defined by the solar azimuth angle and elevation angle.
    The azimuth angle is the counterclockwise angle between the sun and the north direction.
    The elevation angle is the angle between the sun and the horizon.
    The mask is 1 for shaded cells.
    """
    if solar_elevation_angle == 90:
        return np.zeros(dem.arr.shape, dtype=int)
    dem_points = _point_representation_of_dem(_double_resolution_of_raster(dem))
    rotated_dem_points = _rotate_points_around_z_axis(dem_points, solar_azimuth_angle)
    rotated_dem = _raster_representation_of_points_max_z(
        rotated_dem_points, dem.dx, dem.dy
    )
    rotated_mask = _shading_mask_from_sun_at_north_horizon(
        _add_y_gradient(
            rotated_dem, _gradient_from_elevation_angle(solar_elevation_angle)
        )
    )
    point_mask = _raster_values_at_points(
        rotated_mask, rotated_dem_points[0, :], rotated_dem_points[1, :]
    )
    return _unflatten_vector_to_raster_dimensions(
        point_mask, dem.arr.shape[0] * 2, dem.arr.shape[1] * 2
    )


def _rad(degrees: float) -> float:
    return degrees * np.pi / 180


def _fill_nans(arr: np.ndarray, value: float) -> np.ndarray:
    return np.where(np.isnan(arr), value, arr)


def _point_representation_of_dem(dem: Raster) -> np.ndarray:
    """Create a point representation of a DEM.
    The representation XYZ is an arrary of shape (3, N) like [X, Y, Z],
    where each of X, Y, and Z is a 1D array whose length is the number of cells in the dem grid.
    """
    X = np.vstack(
        [dem.transform.c + dem.dx * (np.arange(dem.arr.shape[1]) + 0.5)]
        * dem.arr.shape[0]
    )
    Y = np.vstack(
        [dem.transform.f - dem.dy * (np.arange(dem.arr.shape[0]) + 0.5)]
        * dem.arr.shape[1]
    ).transpose()
    Z = dem.arr
    return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)


def _rotate_points_around_z_axis(XYZ: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points XYZ counterclockwise around the z-axis.

    Args:
        XYZ: Array of shape (3, N) where rows are [X, Y, Z]
        angle: Angle in degrees to rotate counterclockwise around the z-axis

    Returns:
        Array of shape (3, N) where rows are the rotated [X, Y, Z]
    """
    rotation_matrix = np.array(
        [
            [np.cos(_rad(angle)), -np.sin(_rad(angle)), 0],
            [np.sin(_rad(angle)), np.cos(_rad(angle)), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(rotation_matrix, XYZ)


def _raster_representation_of_points_max_z(
    XYZ: np.ndarray,
    dx: float,
    dy: float,
    crs: pyproj.CRS | None = None,
) -> Raster:
    # TODO: change this to use mean Z value of points in each cell
    """Convert a 3D array of XYZ points to a Raster using max Z values per cell.

    Args:
        XYZ: Array of shape (3, N) where rows are [X, Y, Z]
        dx: Grid cell spacing in X direction
        dy: Grid cell spacing in Y direction
        crs: Coordinate reference system. Defaults to None because crs isn't relevant to the
            application of this helper function.

    Returns:
        Raster with array values being the maximum Z value for each grid cell.
        Cells with no points will have NaN values.
    """
    X, Y, Z = XYZ[0, :], XYZ[1, :], XYZ[2, :]

    # Add half a cell to the min and max to ensure the raster covers the points
    # The half-cell buffer will also make it possibole to recover the
    # oringinal raster in a [raster -> points -> raster] round trip.
    x_min, x_max = np.min(X) - dx / 2, np.max(X) + dx / 2
    y_min, y_max = np.min(Y) - dy / 2, np.max(Y) + dy / 2
    n_cols = int(np.ceil((x_max - x_min) / dx)) + 1
    n_rows = int(np.ceil((y_max - y_min) / dy)) + 1

    col_indices = np.floor((X - x_min) / dx).astype(int)
    row_indices = np.floor((y_max - Y) / dy).astype(int)
    # Saturate indices to valid range
    col_indices = np.clip(col_indices, 0, n_cols - 1)
    row_indices = np.clip(row_indices, 0, n_rows - 1)
    flat_indices = row_indices * n_cols + col_indices

    raster_arr = np.full((n_rows, n_cols), np.nan)

    # TODO: review this algorithm
    sort_idx = np.argsort(flat_indices)
    sorted_indices = flat_indices[sort_idx]
    sorted_Z = Z[sort_idx]

    # Find where indices change (group boundaries)
    diff = np.diff(sorted_indices)
    group_starts = np.concatenate(([0], np.where(diff > 0)[0] + 1))
    group_ends = np.concatenate((group_starts[1:], [len(sorted_indices)]))

    # Compute max for each group
    for start, end in zip(group_starts, group_ends):
        idx = sorted_indices[start]
        row = idx // n_cols
        col = idx % n_cols
        raster_arr[row, col] = np.max(sorted_Z[start:end])

    transform = rasterio.Affine(dx, 0.0, x_min, 0.0, -dy, y_max)

    return Raster(arr=raster_arr, transform=transform, crs=crs)


def _add_y_gradient(dem: Raster, gradient: float) -> Raster:
    """Add a y-gradient to a DEM."""
    new_dem = dem.copy()
    n_rows, n_cols = new_dem.arr.shape
    new_dem.arr -= (
        np.vstack([np.arange(n_rows)] * n_cols).transpose().astype(float)
        * gradient
        * new_dem.dy
    )
    return new_dem


def _gradient_from_elevation_angle(elevation_angle: float) -> float:
    """Get the gradient of a slope that parallels an elevation angle.
    Avoid infinity at 90 degrees——will need to handle 90-degree case elsewhere.
    """
    assert 0 <= elevation_angle < 90, "Elevation angle must be in [0, 90) degrees"
    return -np.tan(_rad(elevation_angle))


def _shading_mask_from_sun_at_north_horizon(dem: Raster) -> Raster:
    """Create a mask showing what cells would be shaded by as sun at the north horizon.
    The mask is 1 for shaded cells.
    """
    _, grad_y = dem_to_gradient(dem)
    mask = np.zeros(dem.arr.shape, dtype=int)
    n_rows, _ = dem.arr.shape
    cumulative_max_elevation = _fill_nans(dem.arr[0, :], -np.inf)
    for row_num in range(1, n_rows):
        row_elevations = _fill_nans(dem.arr[row_num, :], -np.inf)
        cumulative_max_elevation = np.maximum(cumulative_max_elevation, row_elevations)
        mask[row_num, row_elevations < cumulative_max_elevation] = 1
    mask[grad_y > 0] = 1  # South facing slope will be shaded
    return dem.with_array(mask)


def _raster_values_at_points(
    raster: Raster, X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """Get the values of a raster at points [X, Y].

    Args:
        raster: Raster
        X: array of shape (N,) where the i-th element is the X coordinate of the i-th point
        Y: array of shape (N,) where the i-th element is the Y coordinate of the i-th point
    Returns:
        Array of shape (N,) where the i-th element is the value of the raster at the point [X[i], Y[i]]
    """
    col_indices = np.floor((X - raster.transform.c) / raster.dx).astype(int)
    row_indices = np.floor((Y - raster.transform.f) / (-raster.dy)).astype(int)
    return raster.arr[row_indices, col_indices]


def _unflatten_vector_to_raster_dimensions(
    vector: np.ndarray, n_rows: int, n_cols: int
) -> np.ndarray:
    """Unflatten a vector to match the dimensions of a raster."""
    return vector.reshape(n_rows, n_cols)


def _double_resolution_of_array(arr: np.ndarray) -> np.ndarray:
    """Double the resolution of an array.
    Every other row and column contains values from the original array,
    and the intermediate rows and columns are means of the original
    surrounding values.
    E.g.:
        If input is [[a, b], [c, d]], output will be:
        [[a, (a+b)/2, b],
         [(a+c)/2, (a+b+c+d)/4, (b+d)/2],
         [c, (c+d)/2, d]].

    Args:
        arr: 2D numpy array of shape (m, n)

    Returns:
        2D numpy array of shape (2m, 2n)

    Notes:
    Because the application for this is to double the resolution of a raster,
    it's important that the output have twice as many rows and columns as the
    input. Otherwise (considering an output with 2m - 1 rows or 2n - 1 columns),
    it would be difficult to preserve the geographic extent of the original raster.
    This means:
        - The output array will always have an even number of rows and columns.
        - Every even (0, 2, ...) row and column (except the last) will alternate between
          an original value and an average of 2 surrounding values.
        - Every odd (1, 3, ...) row and column will be an average of 4 surrounding values.
        - The last row and column will be a copy of the last original row and column.
    """
    m, n = arr.shape
    assert m > 1 and n > 1, (
        "Array must have at least 2 rows and 2 columns to double resolution"
    )
    result = np.zeros((2 * m, 2 * n), dtype=arr.dtype)

    # Place original values at even indices
    result[::2, ::2] = arr

    # Fill in intermediate columns (odd columns, even rows)
    # with average of left and right neighbors
    result[::2, 1::2][:, : n - 1] = (arr[:, :-1] + arr[:, 1:]) / 2
    # Last intermediate column: copy the last original column
    # TODO: is this necessary?
    result[::2, 1::2][:, -1] = arr[:, -1]

    # Fill in intermediate rows (odd rows, even columns)
    # Average of top and bottom neighbors
    result[1::2, ::2][: m - 1, :] = (arr[:-1, :] + arr[1:, :]) / 2
    # Last intermediate row: copy the last original row
    # TODO: is this necessary?
    result[1::2, ::2][-1, :] = arr[-1, :]

    # Fill in intermediate positions (odd rows, odd columns)
    # Average of four neighbors
    # For positions between both rows and columns
    result[1::2, 1::2][: m - 1, : n - 1] = (
        arr[:-1, :-1] + arr[:-1, 1:] + arr[1:, :-1] + arr[1:, 1:]
    ) / 4
    # Right edge: average of two vertical neighbors
    result[1::2, 1::2][: m - 1, -1] = (arr[:-1, -1] + arr[1:, -1]) / 2
    # Bottom edge: average of two horizontal neighbors
    result[1::2, 1::2][-1, : n - 1] = (arr[-1, :-1] + arr[-1, 1:]) / 2
    # Bottom-right corner: copy the last original value
    result[1::2, 1::2][-1, -1] = arr[-1, -1]

    return result


def _double_resolution_of_raster(raster: Raster) -> Raster:
    """Double the resolution of a raster."""
    new_transform = rasterio.Affine(
        raster.dx / 2,
        0.0,
        raster.origin[0],
        0.0,
        -raster.dy / 2,
        raster.origin[1],
    )
    return Raster(
        arr=_double_resolution_of_array(raster.arr),
        transform=new_transform,
        crs=raster.crs,
    )
