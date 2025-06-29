import pytest

from insolation_model.raster import Raster
from tests.test_data.context import dem_path


@pytest.fixture
def dem():
    return Raster.from_tif(dem_path)
