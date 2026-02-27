"""To find solar position as a function of time.
These are adapted from equations in the resources folder.
I chose to ignore time-zone/longitude related complications that are not relevant to seasonal insolation.
"""

import numpy as np

_earth_axial_tilt = 23.44  # in degrees
_day_of_winter_solstice = -10  # December 21st


def get_fractional_year(day_of_year: int, hour: float) -> float:
    """The fractional year is the fraction of one year since the vernal equinox."""
    return (
        ((day_of_year - _day_of_winter_solstice) % 365 + ((hour - 12) / 24)) / 365
    ) % 1


def get_solar_declination(fractional_year: float) -> float:
    """The solar declination is the angle between the sun and the equator.

    Args:
        fractional_year: time in fraction of one year past the winter equinox.

    Returns:
        The solar declination in degrees.
    """
    return -_earth_axial_tilt * np.cos(fractional_year * 2 * np.pi)


def get_solar_hour_angle(solar_minutes: float) -> float:
    """The solar hour angle is the angle between the sun and the local meridian.

    Args:
        solar_minutes: The local solar time in minutes.

    Returns:
        The solar hour angle in degrees.
    """
    return solar_minutes / 4 - 180


def get_solar_unit_direction(
    latitude: float, day_of_year: int, hour: float
) -> np.ndarray:
    """The solar unit direction is the unit vector pointing from an observer on the surface of the earth to the sun."""
    fractional_year = get_fractional_year(day_of_year, hour)
    declination = get_solar_declination(fractional_year)
    solar_minutes = hour * 60
    solar_hour_angle = get_solar_hour_angle(solar_minutes)
    return np.array(
        [
            np.cos(declination * np.pi / 180) * np.sin(solar_hour_angle * np.pi / 180),
            np.cos(latitude * np.pi / 180) * np.sin(declination * np.pi / 180)
            - np.sin(latitude * np.pi / 180)
            * np.cos(declination * np.pi / 180)
            * np.cos(solar_hour_angle * np.pi / 180),
            np.sin(latitude * np.pi / 180) * np.sin(declination * np.pi / 180)
            + np.cos(latitude * np.pi / 180)
            * np.cos(declination * np.pi / 180)
            * np.cos(solar_hour_angle * np.pi / 180),
        ]
    )
