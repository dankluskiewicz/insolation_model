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
    mask = np.zeros(dem.arr.shape, dtype=float)
    n_rows, _ = dem.arr.shape
    cumulative_max_elevation = _fill_nans(dem.arr[0, :], -np.inf)
    for row_num in range(1, n_rows):
        row_elevations = _fill_nans(dem.arr[row_num, :], -np.inf)
        cumulative_max_elevation = np.maximum(cumulative_max_elevation, row_elevations)
        mask[row_num, row_elevations < cumulative_max_elevation] = 1
    mask[grad_y > 0] = 1  # South facing slope will be shaded
    mask[np.isnan(dem.arr)] = np.nan
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
    Every cell in the input array is replaced by a 2x2 block of cells in the output array,
    where all 4 cells in the block have the same value as the input cell.
    """
    # Repeat each row twice, then repeat each column twice
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


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


def _halve_resolution_of_an_array(arr: np.ndarray) -> np.ndarray:
    """Halve the resolution of an array.
    Every cell in the output array is the average of the 4 cells in the input array that it replaces.
    This is the inverse of _double_resolution_of_an_array.
    """
    # the arrays this is applied to are always even-dimensional, and that assumption makes this easier
    m, n = arr.shape
    if m % 2 != 0 or n % 2 != 0:
        raise ValueError(
            "Array must have even number of rows and columns to halve resolution"
        )
    # Reshape to group 2x2 blocks: (m//2, 2, n//2, 2)
    reshaped = arr.reshape(m // 2, 2, n // 2, 2)
    # Take mean along the block dimensions (axes 1 and 3)
    return reshaped.mean(axis=(1, 3))


def _halve_resolution_of_raster(raster: Raster) -> Raster:
    new_transform = rasterio.Affine(
        raster.dx * 2,
        0.0,
        raster.origin[0],
        0.0,
        -raster.dy * 2,
        raster.origin[1],
    )
    return Raster(
        arr=_halve_resolution_of_an_array(raster.arr),
        transform=new_transform,
        crs=raster.crs,
    )
