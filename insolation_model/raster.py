import rasterio
import numpy as np
from pathlib import Path


class Raster:
    def __init__(
        self, arr: np.ndarray, transform: rasterio.Affine, crs: str | None = None
    ):
        self.arr = arr
        self.transform = transform
        self.crs = crs

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
