import pytest
import numpy as np
from insolation_model.geometry.solar import (
    _earth_axial_tilt,
    _day_of_winter_solstice,
    get_fractional_year,
    get_solar_declination,
    get_solar_unit_direction,
)

_day_of_vernal_equinox = _day_of_winter_solstice + 365 / 4
_day_of_summer_solstice = _day_of_winter_solstice + 365 / 2
_day_of_autumnal_equinox = _day_of_winter_solstice + 365 / 4 * 3


def test_get_fractional_year():
    assert get_fractional_year(_day_of_winter_solstice, 12) == 0.0
    assert np.isclose(get_fractional_year(_day_of_winter_solstice, 11), 1.0, atol=1e-3)
    assert np.isclose(
        get_fractional_year(_day_of_winter_solstice + 365, 11), 1.0, atol=1e-3
    )
    assert np.isclose(
        get_fractional_year(_day_of_winter_solstice + 365, 13), 0.0, atol=1e-3
    )


def test_get_solar_declination():
    # declination should be zero at the equinoxes
    assert np.isclose(get_solar_declination(0.25), 0.0, atol=1e-5)
    assert np.isclose(get_solar_declination(0.75), 0.0, atol=1e-5)
    # and should be at the extremes during solstices
    assert np.isclose(get_solar_declination(0.0), -_earth_axial_tilt, atol=1e-5)
    assert np.isclose(get_solar_declination(0.5), _earth_axial_tilt, atol=1e-5)
    assert np.isclose(get_solar_declination(1.0), -_earth_axial_tilt, atol=1e-5)


@pytest.mark.parametrize("latitude", [0, 45, 60, 90])
@pytest.mark.parametrize("day_of_year", [0, 45, 90, 180, 360])
@pytest.mark.parametrize("hour", [0, 9, 12, 15, 24])
def test_solar_unit_direction_has_unit_length(latitude, day_of_year, hour):
    unit_direction = get_solar_unit_direction(latitude, day_of_year, hour)
    assert np.isclose(np.linalg.norm(unit_direction), 1.0, atol=1e-5)


def _is_up(unit_direction):
    return all(
        [
            np.isclose(unit_direction[2], 1.0, atol=1e-5),
            np.isclose(unit_direction[0], 0.0, atol=1e-5),
            np.isclose(unit_direction[1], 0.0, atol=1e-5),
        ]
    )


@pytest.mark.parametrize(
    "day_of_year", [_day_of_vernal_equinox, _day_of_autumnal_equinox]
)
def test_solar_unit_direction_is_up_at_noon_on_equinox_at_equator(day_of_year):
    latitude = 0
    hour = 12
    unit_direction = get_solar_unit_direction(latitude, day_of_year, hour)
    assert _is_up(unit_direction)


@pytest.mark.parametrize(
    "day_of_year", [_day_of_autumnal_equinox, _day_of_vernal_equinox]
)
@pytest.mark.parametrize("hour", [-6, 6])
def test_solar_unit_direction_is_horizontal_at_morning_and_night_at_equator(
    day_of_year, hour
):
    latitude = 0
    unit_direction = get_solar_unit_direction(latitude, day_of_year, hour)
    assert np.isclose(unit_direction[2], 0.0, atol=1e-5)
    assert np.isclose(unit_direction[1], 0.0, atol=1e-2), unit_direction


@pytest.mark.parametrize("latitude", [90, -90])
@pytest.mark.parametrize(
    "day_of_year", [_day_of_autumnal_equinox, _day_of_vernal_equinox]
)
@pytest.mark.parametrize("hour", [0, 2, 3, 15, 24])
def test_solar_unit_direction_is_horizontal_at_poles_during_equinoxes(
    latitude, day_of_year, hour
):
    unit_direction = get_solar_unit_direction(latitude, day_of_year, hour)
    assert np.isclose(unit_direction[2], 0.0, atol=1e-2), unit_direction


@pytest.mark.parametrize(
    ["day_of_year", "latitude"],
    [
        [_day_of_winter_solstice, -_earth_axial_tilt],
        [_day_of_summer_solstice, _earth_axial_tilt],
    ],
)
def test_solar_unit_direction_is_up_at_noon_on_solstice_at_tropics(
    day_of_year, latitude
):
    hour = 12
    unit_direction = get_solar_unit_direction(latitude, day_of_year, hour)
    assert _is_up(unit_direction)
