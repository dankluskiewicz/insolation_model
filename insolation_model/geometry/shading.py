import numpy as np
import rasterio
import pyproj
from ..raster import Raster
from .topography import dem_to_surface_normal_unit_direction


def get_shading_factor(dem: Raster) -> Raster:
    """Get the shading factor for a DEM."""
    surface_normal_unit_direction = dem_to_surface_normal_unit_direction(dem)
    ...


def _rad(degrees: float) -> float:
    return degrees * np.pi / 180


def _point_representation_of_dem(dem: Raster) -> np.ndarray:
    """Convert a DEM to a point representation
    The representation is like [X, Y, Z], where each of X, Y, and Z is a 1D array whose length is the number of cells in the dem grid.
    Each (X[i], Y[i], Z[i]) represents a point on the surface of the dem.
    """
    X = np.vstack(
        [dem.transform.c + dem.dx * np.arange(dem.arr.shape[1])] * dem.arr.shape[0]
    )
    Y = np.vstack(
        [dem.transform.f + dem.dy * np.arange(dem.arr.shape[0])] * dem.arr.shape[1]
    ).transpose()
    Z = dem.arr
    return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)


def _rotate_points_around_z_axis(XYZ: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points XYZ counterclockwise around the z-axis."""
    rotation_matrix = np.array(
        [
            [np.cos(_rad(angle)), -np.sin(_rad(angle)), 0],
            [np.sin(_rad(angle)), np.cos(_rad(angle)), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(rotation_matrix, XYZ)


def _rotate_points_around_x_axis(XYZ: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points XYZ counterclockwise around the x-axis."""
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(_rad(angle)), np.sin(_rad(angle))],
            [0, -np.sin(_rad(angle)), np.cos(_rad(angle))],
        ]
    )
    return np.dot(rotation_matrix, XYZ)
