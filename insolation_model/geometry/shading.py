import numpy as np
from ..raster import Raster
from .topography import dem_to_surface_normal_unit_direction

def get_shading_factor(dem: Raster) -> Raster:
    """Get the shading factor for a DEM."""
    surface_normal_unit_direction = dem_to_surface_normal_unit_direction(dem)
    ...


def _point_representation_of_dem(dem: Raster) -> np.ndarray:
    """Convert a DEM to a point representation."""
    X = dem.transform.c + dem.dx * np.arange(dem.arr.shape[1])
    Y = dem.transform.f + dem.dy * np.arange(dem.arr.shape[0])
    Z = dem.arr
    return np.stack([X, Y, Z], axis=0)


def raster_representation_of_points(points: np.ndarray, dx, dy) -> Raster:
    ...


def _rotate_points_around_z_axis(XYZ: np.ndarray, angle: float) -> np.ndarray:
    """Rotate point data clockwise around the z-axis.
    After rotation, the point data will be oriented such that the positive y-axis is pointing <angle>
    degrees clockwise from North.
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return np.dot(rotation_matrix, XYZ)

def _rotate_points_around_x_axis(XYZ: np.ndarray, angle: float) -> np.ndarray:
    ...


