import pytest
from .test_data.context import _dem_path

from insolation_model.raster import Raster


@pytest.fixture
def dem():
    return Raster.from_tif(_dem_path)
