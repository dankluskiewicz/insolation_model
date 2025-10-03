import numpy as np


def dem_to_surface_normal_unit_direction(dem_array: np.ndarray) -> np.ndarray:
    """Convert a DEM array to a unit direction array."""
    ...


def get_surface_angle_coefficient(surface_normal_unit_direction: np.ndarray, solar_unit_direction: np.ndarray) -> float:
    """Get the coefficient that corrects solar flux for the angle of the ground surface."""
    return np.dot(surface_normal_unit_direction, solar_unit_direction)
