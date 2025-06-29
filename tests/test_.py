def test_loading_dem(dem):
    assert True


def test_raster_bounds_make_sense(dem):
    assert dem.bounds[0] < dem.bounds[2], "xmin should be less than xmax"
    assert dem.bounds[1] < dem.bounds[3], "ymin should be less than ymax"
