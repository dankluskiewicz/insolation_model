import numpy as np
from insolation_model.geometry.solar import (
    _earth_axial_tilt,
    _day_of_winter_solstice,
    get_fractional_year,
    get_solar_declination,
)


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
    assert np.isclose(get_solar_declination(0.0), -_earth_axial_tilt, atol=1e-5)
    assert np.isclose(get_solar_declination(0.5), _earth_axial_tilt, atol=1e-5)
    assert np.isclose(get_solar_declination(1.0), -_earth_axial_tilt, atol=1e-5)
