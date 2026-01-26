import numpy as np
import rasterio
import pyproj
from scipy import ndimage as nd
from ..raster import Raster
from .topography import dem_to_gradient


def get_shading_mask(
    dem: Raster, solar_azimuth_angle: float, solar_elevation_angle: float
) -> np.ndarray:
    """Get the shading mask for a DEM, given a solar position.
    The solar position is the solar azimuth angle and elevation angle.
    The azimuth angle is the counterclockwise angle between the sun and the north direction.
    The elevation angle is the angle between the sun and the horizon.
    The mask is 1 for unshaded cells and 0 for shaded cells. Some cells may have partial shading.
    """
    if solar_elevation_angle == 90:
        return np.ones(dem.arr.shape, dtype=float)
    doubled_dem = _double_resolution_of_raster(dem)
    doubled_dem_points = _point_representation_of_dem(doubled_dem)
    rotated_doubled_dem_points = _rotate_points_around_z_axis(
        doubled_dem_points, solar_azimuth_angle
    )
    rotated_dem = _raster_representation_of_points_mean_z(
        rotated_doubled_dem_points,
        dem.dx,
        dem.dy,
        *_rotate_raster_corners_around_z_axis(dem, solar_azimuth_angle),
        crs=dem.crs,
    )
    rotated_mask = _shading_mask_from_sun_at_north_horizon(
        _add_y_gradient(
            rotated_dem, _gradient_from_elevation_angle(solar_elevation_angle)
        )
    )
    point_mask = _raster_values_at_points(
        rotated_mask, rotated_doubled_dem_points[0, :], rotated_doubled_dem_points[1, :]
    )
    return _halve_resolution_of_an_array(
        _unflatten_vector_to_raster_dimensions(point_mask, *doubled_dem.arr.shape)
    )


def _rad(degrees: float) -> float:
    return degrees * np.pi / 180


def _fill_nans(arr: np.ndarray, value: float) -> np.ndarray:
    return np.where(np.isnan(arr), value, arr)


def _fill_nans_with_nearest_neighbor(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values with the nearest valid value of neighboring cells."""
    invalid = np.isnan(arr)
    if not invalid.any():
        return arr
    indices = nd.distance_transform_edt(
        invalid, return_distances=False, return_indices=True
    )
    return arr[tuple(indices)]


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


def _raster_representation_of_points_mean_z(
    XYZ: np.ndarray,
    dx: float,
    dy: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crs: pyproj.CRS | None = None,
) -> Raster:
    """Convert a 3D array of XYZ points to a Raster using mean of Z values per cell.
    Arg XYZ is an array of shape (3, N) where rows are [X, Y, Z]
    Returns a Raster with array values being the mean Z value for each grid cell.
    Cells with no points will have NaN values.
    """
    X, Y, Z = XYZ[0, :], XYZ[1, :], XYZ[2, :]

    n_cols = int(np.ceil((xmax - xmin) / dx))
    n_rows = int(np.ceil((ymax - ymin) / dy))

    col_indices = np.floor((X - xmin) / dx).astype(int)
    row_indices = np.floor((ymax - Y) / dy).astype(int)

    # Compute mean Z value for each cell using bincount
    # Sum all Z values for each unique flat_index
    flat_indices = row_indices * n_cols + col_indices
    n_cells = n_rows * n_cols
    sums = np.bincount(flat_indices, weights=Z, minlength=n_cells)
    counts = np.bincount(flat_indices, minlength=n_cells)
    means = np.where(counts > 0, sums / counts, np.nan)
    print(f"{n_rows=}, {n_cols=}, {means.shape=}")
    raster_arr = _unflatten_vector_to_raster_dimensions(means, n_rows, n_cols)

    transform = rasterio.Affine(dx, 0.0, xmin, 0.0, -dy, ymax)

    return Raster(arr=raster_arr, transform=transform, crs=crs)


def _rotate_raster_corners_around_z_axis(
    dem: Raster, angle: float
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = dem.bounds
    corners = np.array(
        [[xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax], [0, 0, 0, 0]]
    )
    rotated_corners = _rotate_points_around_z_axis(corners, angle)
    rotated_xmin = np.min(rotated_corners[0, :])
    rotated_xmax = np.max(rotated_corners[0, :])
    rotated_ymin = np.min(rotated_corners[1, :])
    rotated_ymax = np.max(rotated_corners[1, :])
    return rotated_xmin, rotated_ymin, rotated_xmax, rotated_ymax


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
    The mask is 1 for unshaded cells and 0 for shaded cells.
    """
    _, grad_y = dem_to_gradient(dem)
    mask = np.ones(dem.arr.shape, dtype=float)
    n_rows, _ = dem.arr.shape
    cumulative_max_elevation = _fill_nans(dem.arr[0, :], -np.inf)
    for row_num in range(1, n_rows):
        row_elevations = _fill_nans(dem.arr[row_num, :], -np.inf)
        cumulative_max_elevation = np.maximum(cumulative_max_elevation, row_elevations)
        mask[row_num, row_elevations < cumulative_max_elevation] = 0
    mask[grad_y > 0] = 0  # South facing slope will be shaded
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
