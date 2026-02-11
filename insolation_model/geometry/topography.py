import numpy as np
from ..raster import Raster


def dem_to_gradient(dem: Raster) -> np.ndarray:
    grad_y, grad_x = np.gradient(dem.arr, -dem.dy, dem.dx)
    return np.stack([grad_x, grad_y])


def dem_to_surface_normal_unit_direction(dem: Raster) -> np.ndarray:
    """Convert a DEM raster to an array of  unit vectors that are perpendicular to the surface."""
    grad_x, grad_y = dem_to_gradient(dem)
    # For a surface z = f(x, y), an (upward-pointing) normal direction is (-dz/dx, -dz/dy, 1).
    # Normalize to a unit vector at each pixel.
    denom = np.sqrt(1.0 + grad_x**2 + grad_y**2)
    normal_x = -grad_x / denom
    normal_y = -grad_y / denom
    normal_z = 1.0 / denom
    return np.stack([normal_x, normal_y, normal_z])


def _get_dot_product_of_surface_normal_and_solar_unit_directions(
    surface_normal_unit_direction: np.ndarray, solar_unit_direction: np.ndarray
) -> float:
    """Get the dot product of the surface normal unit direction and the solar unit direction.
    Saturate from below at 0 to use this as a coefficient that corrects solar flux for the angle of the ground surface,
    or use it directly to for shaded relief visuals.

    Args:
        surface_normal_unit_direction: The unit vector that is perpendicular to the surface, shape (3, n_rows, n_cols).
        solar_unit_direction: The unit vector that is pointing from the surface to the sun, shape (3,).

    Returns:
        The dot product of the surface normal unit direction and the solar unit direction, shape (n_rows, n_cols).
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
        solar_azimuth: Solar azimuth angle in degrees clockwise from north.
        solar_elevation: Solar elevation angle in degrees above the horizon.

    Returns:
        Raster whose values are the dot product of the surface normal unit direction and the solar unit direction, shape (n_rows, n_cols).
    """
    surface_normal_unit_direction = dem_to_surface_normal_unit_direction(dem)
    solar_unit_direction = _get_solar_unit_direction_from_angular_position(
        solar_azimuth, solar_elevation
    )
    surface_angle_coefficient = (
        _get_dot_product_of_surface_normal_and_solar_unit_directions(
            surface_normal_unit_direction, solar_unit_direction
        )
    )
    return dem.with_array(surface_angle_coefficient)


def dem_to_topographic_flux_coefficient(
    dem: Raster,
    solar_azimuth: float = 315,
    solar_elevation: float = 45,
) -> Raster:
    """Compute the coefficient that corrects solar flux for the angle of the ground surface.

    Returns an raster whose values are in [0, 1]. 0 is shaded. 1 is fully illuminated.
    Shape (n_rows, n_cols)

    Args:
        dem: Digital elevation model raster.
        solar_azimuth: Solar azimuth angle in degrees clockwise from north.
        solar_elevation: Solar elevation angle in degrees above the horizon.
    """
    hillshade = dem_to_hillshade(dem, solar_azimuth, solar_elevation)
    return hillshade.with_array(np.maximum(0, hillshade.arr))
