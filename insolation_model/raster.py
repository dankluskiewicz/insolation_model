"""tools to read and manipulate raster."""

from pathlib import Path
import rasterio
from rasterio import warp
import numpy as np


class Raster:
    def __init__(
        self, arr: np.ndarray, transform: rasterio.Affine, crs: str | None = None
    ):
        self.arr = arr
        self.transform = transform
        self.crs = crs

    @property
    def dx(self) -> float:
        return self.transform.a

    @property
    def dy(self) -> float:
        return self.transform.e

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.transform.c, self.transform.f])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """xmin, ymin, xmax, ymax."""
        return (
            self.origin[0],
            self.origin[1] + self.dy * self.arr.shape[0],
            self.origin[0] + self.dx * self.arr.shape[1],
            self.origin[1],
        )

    @classmethod
    def from_tif(cls, path: Path) -> "Raster":
        with rasterio.open(path) as src:
            arr = src.read()
            if arr.ndim > 2:
                if arr.shape[0] == 1:
                    arr = arr[0, :, :]
                else:
                    raise Exception("This class can't handle multiband data")

            return cls(
                arr=arr,
                transform=src.transform,
                crs=src.crs,
            )

    def reproject(self, new_crs: str, dx=None) -> "Raster":
        """Reproject self to a new CRS with similar spatial resolution and coverage."""
        new_transform, new_width, new_height = warp.calculate_default_transform(
            self.crs,
            new_crs,
            self.arr.shape[1],
            self.arr.shape[0],
            *self.bounds,
        )

        if new_width is None or new_height is None:
            raise ValueError("Could not calculate new raster dimensions")

        new_arr = np.empty((new_height, new_width))

        warp.reproject(
            self.arr,
            new_arr,
            src_transform=self.transform,
            dst_transform=new_transform,
            src_crs=self.crs,
            dst_crs=new_crs,
            resampling=warp.Resampling.nearest,
        )

        return Raster(new_arr, new_transform, new_crs)
