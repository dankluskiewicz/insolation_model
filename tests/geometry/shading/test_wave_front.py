import pytest
import numpy as np
from insolation_model.geometry.shading import _make_wave_front


@pytest.mark.parametrize("azimuth", [0, 10, 25, 40, 45])
@pytest.mark.parametrize("n_rows", [3, 9, 23, 117])
@pytest.mark.parametrize("n_cols", [3, 11, 17, 113])
def test_wave_front_covers_entire_raster(azimuth, n_rows, n_cols):
    Fi, Fj = _make_wave_front(n_rows, n_cols, azimuth)
    assert set(np.floor(Fi).flatten().astype(int)) >= set(range(n_rows))
    assert set(np.floor(Fj).flatten().astype(int)) >= set(range(n_cols))
    # also check for reasonable economy
    front_length, front_width = Fi.shape
    assert front_length * front_width < (np.max([n_rows, n_cols]) + 1) ** 2 * 4
