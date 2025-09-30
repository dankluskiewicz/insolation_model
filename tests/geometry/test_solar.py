import numpy as np
from insolation_model.geometry.solar import (
    get_fractional_year,
    get_solar_declination,
    _earth_axial_tilt,
    _day_of_vernal_equinox,
)


def test_fractional_year():
    assert get_fractional_year(_day_of_vernal_equinox, 12) == 0.0
    assert np.isclose(get_fractional_year(_day_of_vernal_equinox, 11), 1.0, atol=1e-3)
    assert np.isclose(
        get_fractional_year(_day_of_vernal_equinox + 365, 11), 1.0, atol=1e-3
    )
    assert np.isclose(
        get_fractional_year(_day_of_vernal_equinox + 365, 13), 0.0, atol=1e-3
    )


def test_solar_declination():
    assert np.isclose(get_solar_declination(0.0), 0.0, atol=1e-5)
    assert np.isclose(get_solar_declination(0.5), 0.0, atol=1e-5)
    assert np.isclose(get_solar_declination(1.0), 0.0, atol=1e-5)
    assert np.isclose(get_solar_declination(0.25), _earth_axial_tilt, atol=1e-5)
    assert np.isclose(get_solar_declination(0.75), -_earth_axial_tilt, atol=1e-5)
