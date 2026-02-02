import numpy as np
import rasterio
from scipy import ndimage as nd

from ..raster import Raster


def make_shade_mask(
    dem: Raster, solar_azimuth_angle: float, solar_elevation_angle: float
) -> np.ndarray: ...


def _make_shade_mask_from_horizontal_wave_front(
    dem: Raster,
    wave_front_azimuth_angle: float,
) -> np.ndarray:
    Fi, Fj = _make_wave_front(*dem.arr.shape, wave_front_azimuth_angle)
    Fi_indices, Fj_indices, Fvalues, valid_indices_on_front = (
        _get_raster_values_on_front(dem, Fi, Fj)
    )
    F_cummax = np.maximum.accumulate(Fvalues, axis=0)
    F_mask = (Fvalues < F_cummax).astype(int)
    front_vector_i = Fi_indices[valid_indices_on_front]
    front_vector_j = Fj_indices[valid_indices_on_front]
    front_vector_mask = F_mask[valid_indices_on_front]

    unique_pairs, mean_mask_on_front = _mean_over_indices(
        front_vector_i, front_vector_j, front_vector_mask
    )

    shading_mask = np.zeros_like(dem.arr)
    shading_mask[*unique_pairs] = mean_mask_on_front
    return shading_mask


def _find_wave_front_origin(
    raster_n_rows: int,
    azimuth: float,
) -> np.ndarray:
    """Find the origin of a wave front that will cover a raster.

    Args:
        raster_n_rows: The number of rows in the raster.
        azimuth: The azimuth angle of the wave front.

    Returns:
        The origin of the wave front as a an array of shape (2,).
    """
    return (
        raster_n_rows
        * np.sin(_rad(azimuth))
        * np.array([np.sin(_rad(azimuth)), -np.cos(_rad(azimuth))])
    )


def _find_wave_front_width(
    wave_front_origin: tuple[float, float],
    azimuth: float,
    raster_n_cols: int,
) -> int:
    L1 = np.hypot(
        *wave_front_origin
    )  # distance from wave-front origin to raster origin
    L2 = (raster_n_cols) * np.cos(
        _rad(azimuth)
    )  # distance from raster origin to the upper-right corner of the wave front
    return L1 + L2


def _find_wave_front_front_length(
    wave_front_origin: tuple[float, float],
    raster_n_rows: int,
    raster_n_cols: int,
    azimuth: float,
) -> int:
    raster_bl_corner = np.array([raster_n_rows, 0])
    L3 = np.hypot(
        *(raster_bl_corner - wave_front_origin)
    )  # distance from wave-front origin to the bottom-left corner of the raster
    L4 = (
        (raster_n_cols) * np.sin(_rad(azimuth))
    )  # distance from the bottom-left corner of the raster to the bottom-left corner of the wave front
    return L3 + L4


def _make_wave_front(
    raster_n_rows: int,
    raster_n_cols: int,
    azimuth: float,
    packet_spacing: int = 1 / np.sqrt(2),
    front_spacing: int = 1 / np.sqrt(2),
) -> tuple[np.ndarray, np.ndarray]:
    """Find the discretized locations for a wave front that will cover a raster.
    For this purpose, locations are in pixel space.
    (0, 0) is the raster origin. (1, 1) is the bottom-right corner of the upper-left pixel.

    Args:
        raster_n_rows: The number of rows in the raster.
        raster_n_cols: The number of columns in the raster.
        azimuth: The azimuth angle of the wave front.
        packet_spacing: The spacing between points in the same front (spacing othorgonal to azimuth).
        front_spacing: The spacing between fronts in the wave front (spacing along azimuth).

    Returns:
        Fi: The i-coordinates of the wave front according to the raster array axes.
            i.e. Fi[0, 0] is the horizontal distance between the wave-front and raster origins
            in units of pixels.
        Fj: The j-coordinates of the wave front according to the raster array axes.
    """
    if (azimuth < 0) or (azimuth > 45):
        raise ValueError(
            "Azimuth angle must be between 0 and 45 degrees for make_wave_front."
        )
    wave_front_origin = _find_wave_front_origin(raster_n_rows, azimuth)
    n_packets = int(
        np.ceil(
            _find_wave_front_width(wave_front_origin, azimuth, raster_n_cols)
            / packet_spacing
        )
    )
    n_fronts = int(
        np.ceil(
            _find_wave_front_front_length(
                wave_front_origin, raster_n_rows, raster_n_cols, azimuth
            )
            / front_spacing
        )
    )

    hps = packet_spacing * np.cos(
        _rad(azimuth)
    )  # horizontal packet spacing (in pixels)
    vps = packet_spacing * np.sin(_rad(azimuth))  # vertical packet spacing (in pixels)
    vfs = front_spacing * np.cos(_rad(azimuth))  # vertical front spacing (in pixels)
    hfs = front_spacing * np.sin(_rad(azimuth))  # horizontal front spacing (in pixels)

    i0, j0 = wave_front_origin
    ii0 = i0 - np.arange(n_packets) * vps
    jj0 = j0 + np.arange(n_packets) * hps

    Fi = np.outer(np.ones(n_fronts), ii0) + np.outer(
        np.arange(n_fronts), np.ones(n_packets) * vfs
    )
    Fj = np.outer(np.ones(n_fronts), jj0) + np.outer(
        np.arange(n_fronts), np.ones(n_packets) * hfs
    )
    return Fi, Fj


def _get_raster_values_on_front(
    raster: Raster,
    Fi: np.ndarray,
    Fj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the values of a raster on a wave front.
    Also returns intermediate data that are necessary to compute a shading mask.

    Args:
        raster: The raster to get the values from.
        Fi: The i-coordinates of the wave front according to the raster array axes.
        Fj: The j-coordinates of the wave front according to the raster array axes.

    Returns:
        Fi_indices: The i-coordinates of the wave front rounded down to the nearest integer.
        Fj_indices: The j-coordinates of the wave front rounded down to the nearest integer.
        Fvalues: The values of the raster on the wave front.
        valid_indices_on_front: The indices of the wave front that are inside the raster.
    """
    n_rows, n_cols = raster.arr.shape
    Fi_indices = np.floor(Fi).astype(int)
    Fj_indices = np.floor(Fj).astype(int)
    valid_indices_on_front = (
        (Fi_indices >= 0)
        & (Fj_indices >= 0)
        & (Fi_indices < n_rows)
        & (Fj_indices < n_cols)
    )
    Fvalues = 0 * Fi - 9999
    Fvalues[valid_indices_on_front] = raster.arr[
        Fi_indices[valid_indices_on_front], Fj_indices[valid_indices_on_front]
    ]
    return Fi_indices, Fj_indices, Fvalues, valid_indices_on_front


def _mean_over_indices(
    ii: np.ndarray, jj: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Compute the mean of values for all unique pairs of indices in ii x jj."""
    # TODO: test this
    pairs = np.column_stack((ii, jj))
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique_pairs.T, sums / counts


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
