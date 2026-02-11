import numpy as np
import rasterio
from insolation_model.raster import Raster


def test_loading_raster(dem):
    assert True


def test_raster_bounds_make_sense(dem):
    assert dem.bounds[0] < dem.bounds[2], "xmin should be less than xmax"
    assert dem.bounds[1] < dem.bounds[3], "ymin should be less than ymax"


def test_raster_reprojection(dem):
    # this test won't make sense if input crs is same as output crs
    assert dem.crs.is_geographic
    new_crs = "EPSG:26910"
    new_raster = dem.reproject(new_crs)
    assert new_raster.crs == new_crs

    resurrected_raster = new_raster.reproject(dem.crs)
    assert dem.crs == resurrected_raster.crs
    assert all(np.isclose(dem.transform, resurrected_raster.transform, rtol=1e-2))


def test_in_utm(dem):
    assert dem.crs.is_geographic
    dem_in_utm = dem.in_utm()
    assert dem_in_utm.crs.is_projected
    assert dem.crs.to_epsg() == 4326  # I know this is the correct utm zone


def test_from_tif_converts_nodata_to_nan_and_reproject_preserves_nan(tmp_path):
    path = tmp_path / "tiny_nodata.tif"
    arr = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(arr, 1)

    r = Raster.from_tif(path)
    assert np.isnan(r.arr[0, 1])

    r2 = r.reproject("EPSG:26910")
    assert np.isnan(r2.arr).any()
