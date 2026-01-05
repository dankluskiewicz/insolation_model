import numpy as np
import rasterio
import pyproj
from ..raster import Raster


def get_shading_factor(dem: Raster) -> Raster:
    """Get the shading factor for a DEM."""
    ...


def _rad(degrees: float) -> float:
    return degrees * np.pi / 180


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
        XYZ: Array of shape (3, N) where columns are [X, Y, Z]
        angle: Angle in degrees to rotate counterclockwise around the z-axis

    Returns:
        Array of shape (3, N) where columns are the rotated [X, Y, Z]
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
    """Convert a 3D array of XYZ points to a Raster using max Z values per cell.

    Args:
        XYZ: Array of shape (3, N) where columns are [X, Y, Z]
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
    # oringinal raster in a raster -> points -> raster round trip.
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

    # Use a more efficient approach: sort by index, then group and take max
    # This avoids the loop and is much faster for large datasets
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
