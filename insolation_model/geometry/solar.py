"""To find solar position as a function of time.
These are adapted from the equations in the "Solar Equations" PDF in the resources folder.
I chose to ignore time-zone/longitude related complications that are not relevant to seasonal insolation.
"""

import numpy as np

_earth_axial_tilt = 23.44 * np.pi / 180  # in radians
_day_of_vernal_equinox = 79  # March 20th


def get_fractional_year(day_of_year: int, hour: float) -> float:
    """The fractional year is the fraction of one year since the vernal equinox."""
    return (
        ((day_of_year - _day_of_vernal_equinox) % 365 + ((hour - 12) / 24)) / 365
    ) % 1


def get_solar_declination(fractional_year: float) -> float:
    """The solar declination is the angle between the sun and the equator.

    Args:
        fractional_year: time in fraction of one year.

    Returns:
        The solar declination in radians.
    """
    return _earth_axial_tilt * np.sin(fractional_year * 2 * np.pi)


def get_solar_hour_angle(solar_minutes: float) -> float:
    """The solar hour angle is the angle between the sun and the local meridian.

    Args:
        solar_minutes: The local solar time in minutes.

    Returns:
        The solar hour angle in radians.
    """
    return (solar_minutes / 4 - 180) * np.pi / 180


def get_solar_zenith_angle(
    latitude: float, declination: float, solar_hour_angle: float
) -> float:
    """The solar zenith angle is the angle of the sun from vertical.

    Args:
        latitude: in degrees.
        declination: in radians.
        solar_hour_angle: in radians.
    """
    return np.arccos(
        np.sin(latitude * np.pi / 180) * np.sin(declination)
        + np.cos(latitude * np.pi / 180)
        * np.cos(declination)
        * np.cos(solar_hour_angle)
    )


def get_solar_azimuth(
    latitude: float, declination: float, solar_hour_angle: float
) -> float:
    """The solar azimuth angle is the angle of the sun clockwise from north.

    Args:
        latitude: in degrees.
        declination: in degrees.
        solar_hour_angle: in degrees.
    """
    zenith_angle = get_solar_zenith_angle(latitude, declination, solar_hour_angle)
    return np.pi + np.arccos(
        (np.sin(latitude * np.pi / 180) * np.cos(zenith_angle) - np.sin(declination))
        / (np.cos(latitude * np.pi / 180) * np.sin(zenith_angle))
    )


def get_solar_position(
    latitude: float, day_of_year: int, hour: float
) -> tuple[float, float]:
    """Find the solar position for a given latitude and fractional year.

    Args:
        latitude: in degrees.
        day_of_year: in days.
        hour: in hours.

    Returns:
        The solar elevation and azimuth angles in radians.
    """
    fractional_year = get_fractional_year(day_of_year, hour)
    declination = get_solar_declination(fractional_year)
    solar_minutes = hour * 60
    solar_hour_angle = get_solar_hour_angle(solar_minutes)
    zenith_angle = get_solar_zenith_angle(latitude, declination, solar_hour_angle)
    azimuth_angle = get_solar_azimuth(latitude, declination, solar_hour_angle)
    return zenith_angle, azimuth_angle
