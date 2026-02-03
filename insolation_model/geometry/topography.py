import numpy as np
from ..raster import Raster


def dem_to_gradient(dem: Raster) -> np.ndarray:
    grad_y, grad_x = np.gradient(dem.arr, -dem.dy, dem.dx)
    return np.stack([grad_x, grad_y])


def dem_to_surface_normal_unit_direction(dem: Raster) -> np.ndarray:
    """Convert a DEM raster to an array of  unit vectors that are perpendicular to the surface."""
    vectorized_gradient_function = np.vectorize(
        _gradient_vector_to_surface_normal_unit_direction
    )
    return np.array(vectorized_gradient_function(*dem_to_gradient(dem)))


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


def _get_surface_angle_insolation_coefficient(
    surface_normal_unit_direction: np.ndarray, solar_unit_direction: np.ndarray
) -> float:
    """Get the coefficient that corrects solar flux for the angle of the ground surface.

    Args:
        surface_normal_unit_direction: The unit vector that is perpendicular to the surface, shape (3, n_rows, n_cols).
        solar_unit_direction: The unit vector that is pointing from the surface to the sun, shape (3,).

    Returns:
        The coefficient that corrects solar flux for the angle of the ground surface, shape (n_rows, n_cols).
    """
    return (
        surface_normal_unit_direction * solar_unit_direction[:, np.newaxis, np.newaxis]
    ).sum(axis=0)


def _get_solar_unit_direction_from_angular_position(
    solar_azimuth: float, solar_elevation: float
) -> np.ndarray:
    """Get the solar unit direction from an angular position."""
    return np.array(
        [
            np.cos(np.radians(solar_elevation)) * np.sin(np.radians(solar_azimuth)),
            np.cos(np.radians(solar_elevation)) * np.cos(np.radians(solar_azimuth)),
            np.sin(np.radians(solar_elevation)),
        ]
    )


def dem_to_hillshade(
    dem: Raster,
    solar_azimuth: float = 315,
    solar_elevation: float = 45,
) -> Raster:
    """Compute the coefficient that corrects solar flux for the angle of the ground surface.

    This is the dot product of the surface normal unit direction and the solar unit direction.
    It's similar to, but not exactly the conventional interpretation of "shaded relief".
    Saturate from below at 0 to use this as a coefficient that corrects solar flux for the angle of the ground surface,
    or use it directly to for shaded relief visuals.

    Args:
        dem: Digital elevation model raster.
        solar_azimuth: Light direction in degrees clockwise from north.
        solar_elevation: Light angle in degrees above the horizon.

    Returns:
        Raster whose values are the dot product of the surface normal unit direction and the solar unit direction, shape (n_rows, n_cols).
    """
    surface_normal_unit_direction = dem_to_surface_normal_unit_direction(dem)
    solar_unit_direction = _get_solar_unit_direction_from_angular_position(
        solar_azimuth, solar_elevation
    )
    surface_angle_coefficient = _get_surface_angle_insolation_coefficient(
        surface_normal_unit_direction, solar_unit_direction
    )
    return dem.with_array(surface_angle_coefficient)
