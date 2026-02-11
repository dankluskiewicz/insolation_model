"""tools to read and manipulate rasters."""

import warnings
from pathlib import Path
import rasterio
from rasterio import warp
import numpy as np
import pyproj


class Raster:
    def __init__(self, arr: np.ndarray, transform: rasterio.Affine, crs: pyproj.CRS):
        self.arr = arr.astype(float)
        self.transform = transform
        self.crs = crs
        if self.dx != self.dy:
            raise ValueError("this package has only been tested for square pixels")

    @property
    def dx(self) -> float:
        return self.transform.a

    @property
    def dy(self) -> float:
        return -self.transform.e

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.transform.c, self.transform.f])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """xmin, ymin, xmax, ymax."""
        return (
            self.origin[0],
            self.origin[1] - self.dy * self.arr.shape[0],
            self.origin[0] + self.dx * self.arr.shape[1],
            self.origin[1],
        )

    @classmethod
    def from_tif(cls, path: Path) -> "Raster":
        with rasterio.open(path) as src:
            # Read as a masked array so dataset nodata gets converted to a mask.
            arr = src.read(masked=True)
            if arr.ndim > 2:
                if arr.shape[0] == 1:
                    arr = arr[0, :, :]
                else:
                    raise Exception("This class can't handle multiband data")

            # Convert nodata mask to NaNs for downstream numeric ops.
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.astype(float).filled(np.nan)

            return cls(
                arr=arr,
                transform=src.transform,
                crs=_make_pyproj_crs(src.crs),
            )

    def reproject(self, new_crs: str | pyproj.CRS, dx=None) -> "Raster":
        """Reproject self to a new CRS with similar spatial resolution and coverage."""
        new_crs = _make_pyproj_crs(new_crs)
        new_transform, new_width, new_height = warp.calculate_default_transform(
            self.crs,
            new_crs,
            self.arr.shape[1],
            self.arr.shape[0],
            *self.bounds,
        )
        if new_width is None or new_height is None:
            raise ValueError("Could not calculate new raster dimensions")

        new_arr = np.full((new_height, new_width), np.nan, dtype=float)
        src_nodata = np.nan if np.isnan(self.arr).any() else None

        warp.reproject(
            self.arr,
            new_arr,
            src_transform=self.transform,
            dst_transform=new_transform,
            src_crs=self.crs,
            dst_crs=new_crs,
            resampling=warp.Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
            init_dest_nodata=True,
        )

        return Raster(new_arr, new_transform, new_crs)

    def in_utm(self) -> "Raster":
        """Reproject self to the most appropriate UTM zone."""
        utm_zone = _get_utm_zone(self)
        return self.reproject(utm_zone)

    def copy(self: "Raster") -> "Raster":
        return Raster(self.arr.copy(), self.transform, self.crs)

    def with_array(self, arr: np.ndarray) -> "Raster":
        return Raster(arr, self.transform, self.crs)


def _make_pyproj_crs(crs: str | pyproj.CRS) -> pyproj.CRS:
    if isinstance(crs, str):
        return pyproj.CRS.from_user_input(crs)
    return crs


def _get_utm_zone(raster):
    aoi = pyproj.aoi.AreaOfInterest(
        west_lon_degree=raster.bounds[0],
        south_lat_degree=raster.bounds[1],
        east_lon_degree=raster.bounds[2],
        north_lat_degree=raster.bounds[3],
    )
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84", area_of_interest=aoi
    )
    if len(utm_crs_list) > 2:
        warnings.warn(f"input raster spans {len(utm_crs_list)} UTM zones")
    return pyproj.CRS.from_epsg(utm_crs_list[0].code)
