import numpy as np
import rasterio
from scipy import ndimage as nd

from ..raster import Raster


def get_shading_mask(
    dem: Raster, solar_azimuth_angle: float, solar_elevation_angle: float
) -> np.ndarray: ...


def _rad(degrees: float) -> float:
    return degrees * np.pi / 180


def _fill_nans(arr: np.ndarray, value: float) -> np.ndarray:
    return np.where(np.isnan(arr), value, arr)


def _fill_nans_with_nearest_neighbor(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values with the nearest valid value of neighboring cells."""
    invalid = np.isnan(arr)
    if not invalid.any():
        return arr
    indices = nd.distance_transform_edt(
        invalid, return_distances=False, return_indices=True
    )
    return arr[tuple(indices)]


def _gradient_for_slope_that_parallels_solar_elevation(
    elevation_angle: float, azimuth_angle: float
) -> float:
    if elevation_angle <= 0:
        raise ValueError("Elevation angle must be greater than 0")
    if elevation_angle >= 90:
        raise ValueError("Elevation angle must be less than 90")
    grad_x = np.sin(_rad(azimuth_angle)) * np.tan(_rad(elevation_angle))
    grad_y = np.cos(_rad(azimuth_angle)) * np.tan(_rad(elevation_angle))
    return grad_x, grad_y


def make_dem_with_gradients(
    grad_x: float,
    grad_y: float,
    dx: float,
    dy: float,
    n_rows: int = 4,
    n_cols: int = 5,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Raster:
    """Create a test DEM with prescribed gradients."""
    transform = rasterio.Affine(dx, 0, origin_x, 0, -dy, origin_y)
    return Raster(
        arr=(
            np.vstack([np.arange(n_cols)] * n_rows) * grad_x * dx
            - np.vstack([np.arange(n_rows)] * n_cols).transpose() * grad_y * dy
        ),
        transform=transform,
        crs=rasterio.crs.CRS.from_epsg(4326),
    )
