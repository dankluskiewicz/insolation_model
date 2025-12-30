import numpy as np
from typing import cast
from ..raster import Raster


def dem_to_gradient(dem: Raster) -> np.ndarray:
    grad_y, grad_x = np.gradient(dem.arr, dem.dy, dem.dx)
    return np.stack([grad_x, grad_y])


def dem_to_surface_normal_unit_direction(dem: Raster) -> np.ndarray:
    """Convert a DEM raster to an array of  unit vectors that are perpendicular to the surface."""
    vectorized_gradient_function = np.vectorize(
        gradient_vector_to_surface_normal_unit_direction
    )
    return np.array(vectorized_gradient_function(*dem_to_gradient(dem)))


def get_surface_angle_coefficient(
    surface_normal_unit_direction: np.ndarray, solar_unit_direction: np.ndarray
) -> float:
    """Get the coefficient that corrects solar flux for the angle of the ground surface."""
    return cast(float, np.dot(surface_normal_unit_direction, solar_unit_direction))


def _gradient_vector_to_surface_normal_unit_direction(
    grad_x: float, grad_y: float
) -> tuple[float, float, float]:
    """Find the unit vector that is perpendicular to the surface gradient at a single point."""
    slope = np.arctan(np.hypot(grad_x, grad_y))
    norm_horizontal_component = np.sin(slope)
    return (
        np.sign(grad_x)
        * np.sqrt(norm_horizontal_component**2 / (1 + (grad_y / grad_x) ** 2))
        if grad_x != 0
        else 0,
        np.sign(grad_y)
        * np.sqrt(norm_horizontal_component**2 / (1 + (grad_x / grad_y) ** 2))
        if grad_y != 0
        else 0,
        np.cos(slope),
    )  # norm_x, norm_y, norm_z