import numpy as np
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
