import numpy as np


def test_loading_raster(dem):
    assert True


def test_raster_bounds_make_sense(dem):
    assert dem.bounds[0] < dem.bounds[2], "xmin should be less than xmax"
    assert dem.bounds[1] < dem.bounds[3], "ymin should be less than ymax"


def test_raster_reprojection(dem):
    # this test won't make sense if input crs is same as output crs
    assert dem.crs == "EPSG:4326"
    new_crs = "EPSG:26910"
    new_raster = dem.reproject(new_crs)
    assert new_raster.crs == new_crs

    resurrected_raster = new_raster.reproject(dem.crs)
    assert dem.crs == resurrected_raster.crs
    assert all(np.isclose(dem.transform, resurrected_raster.transform, rtol=1e-2))
