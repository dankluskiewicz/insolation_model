import numpy as np

from insolation_model.raster import Raster
from insolation_model.geometry.topography import dem_to_hillshade
from insolation_model.geometry.shading import make_shade_mask


def _dem_to_topographic_flux_coefficient(
    dem: Raster,
    solar_azimuth: float = 315,
    solar_elevation: float = 45,
) -> np.ndarray:
    """Compute the coefficient that corrects solar flux for the angle of the ground surface.

    Returns an array whose values are in [0, 1]. 0 is shaded. 1 is fully illuminated.

    Args:
        dem: Digital elevation model raster.
        solar_azimuth: Solar azimuth angle in degrees clockwise from north.
        solar_elevation: Solar elevation angle in degrees above the horizon.
    """
    hillshade = dem_to_hillshade(dem, solar_azimuth, solar_elevation)
    return np.maximum(0, hillshade.arr)


def insolation_coefficient(
    dem: Raster,
    solar_azimuth: float = 315,
    solar_elevation: float = 45,
) -> np.ndarray:
    """Compute the coefficient that corrects solar flux for the angle of the ground surface and
    topographic shading.

    Returns an array whose values are in [0, 1].
    """
    print("*(*(*))" * 30)
    return _dem_to_topographic_flux_coefficient(dem, solar_azimuth, solar_elevation) * (
        1 - make_shade_mask(dem, solar_azimuth, solar_elevation)
    )
