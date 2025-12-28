import numpy as np
from typing import cast
from ..raster import Raster


def dem_to_gradient(dem: Raster) -> np.ndarray:
    """Convert a DEM array to a gradient array.

    Returns:
        Stacked gradient arrays with shape (2, height, width) where
        the first element is the y-gradient and the second is the x-gradient.
    """
    grad_y, grad_x = np.gradient(dem.arr, dem.dy, dem.dx)
    return np.stack([grad_x, grad_y])


def dem_to_surface_normal_unit_direction(dem_array: np.ndarray) -> np.ndarray:
    """Convert a DEM array to a unit direction array."""
    raise NotImplementedError


def get_surface_angle_coefficient(surface_normal_unit_direction: np.ndarray, solar_unit_direction: np.ndarray) -> float:
    """Get the coefficient that corrects solar flux for the angle of the ground surface."""
    return cast(float, np.dot(surface_normal_unit_direction, solar_unit_direction))
