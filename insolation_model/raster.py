"""tools to read and manipulate raster."""

from pathlib import Path
import rasterio
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
