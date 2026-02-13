import numpy as np
from scipy import ndimage as nd

from ..raster import Raster


## According to my empirical tests, these parameters are small enough to prevent most
## artifacts in the shading mask related to grid rotations that arise in the shading algorithm.

_packet_spacing = 1 / 2.5
_front_spacing = 1 / 2.5


def make_shade_mask(
    dem: Raster, solar_azimuth_angle: float, solar_elevation_angle: float
) -> np.ndarray:
    if not (0 <= solar_elevation_angle <= 90):
        raise ValueError("Solar elevation angle must be between 0 and 90 degrees")
    if not (0 <= solar_azimuth_angle <= 360):
        raise ValueError("Solar azimuth angle must be between 0 and 360 degrees")
    if solar_elevation_angle == 90:
        return np.zeros(dem.arr.shape, dtype=int)
    dem_with_added_gradient = _add_gradient_to_dem(
        dem,
        *(
            -_gradient_for_slope_that_parallels_solar_elevation(
                solar_elevation_angle, solar_azimuth_angle
            )
        ),
    )
    wave_front_theta = -solar_azimuth_angle % 360
    rotation_angle = 90 * (wave_front_theta // 90)
    rotated_wave_front_theta = (wave_front_theta - rotation_angle) % 360
    rotated_dem_with_added_gradient_array = _rotate_array(
        dem_with_added_gradient.arr, rotation_angle
    )
    return _rotate_array(
        _make_shade_mask_from_horizontal_wave_front(
            rotated_dem_with_added_gradient_array,
            rotated_wave_front_theta,
        ),
        -rotation_angle % 360,
    )
    # if (315 <= solar_azimuth_angle <= 360) or (solar_azimuth_angle == 0):
    #     return _make_shade_mask_from_horizontal_wave_front(
    #         dem_with_added_gradient,
    #         -solar_azimuth_angle % 360,
    #     )
    raise ValueError(f"Solar azimuth angle {solar_azimuth_angle} is not supported")


def _make_shade_mask_from_horizontal_wave_front(
    arr: np.ndarray,
    wave_front_theta: float,
) -> np.ndarray:
    """Make a shade mask that indicates which cells of dem would be shaded from the sun at
    zero elevation angle and at a solar azimuth angle -wave_front_theta.
    wave_front_theta is the angle of the sun in degrees counterclockwise from North,
    which happens to be opposite the convention I've adopted for solar azimuth angles
    called "solar_azimuth_angle".

    This function exists because it's possible to compute shading for sun with a nonzero
    elevation angle by manipulating the gradient of a DEM in a direction parallel to the
    solar azimuth, followed by computing shading for sun with zero elevation angle. See
    make_shade_mask.

    Args:
        arr: a DEM array
        wave_front_theta: The angle of the sun in degrees counterclockwise from North.

    Returns:
        A shade mask where 1 indicates shaded and 0 indicates not shaded.
    """
    Fi, Fj = _make_wave_front(*arr.shape, wave_front_theta)
    Fi_indices, Fj_indices, Fvalues, valid_indices_on_front = (
        _get_array_values_on_front(arr, Fi, Fj)
    )
    F_cummax = np.maximum.accumulate(Fvalues, axis=0)
    F_mask = (Fvalues < F_cummax).astype(int)
    front_vector_i = Fi_indices[valid_indices_on_front]
    front_vector_j = Fj_indices[valid_indices_on_front]
    front_vector_mask = F_mask[valid_indices_on_front]

    unique_pairs, mean_mask_on_front = _mean_over_indices(
        front_vector_i, front_vector_j, front_vector_mask
    )

    shading_mask = np.zeros_like(arr)
    shading_mask[*unique_pairs] = (mean_mask_on_front == 1).astype(int)
    return shading_mask


def _find_wave_front_origin(
    raster_n_rows: int,
    theta: float,
) -> np.ndarray:
    """Find the origin of a wave front that will cover a raster.

    Args:
        raster_n_rows: The number of rows in the raster.
        theta: The azimuth angle of the wave front in degrees counterclockwise from North.
               This is called "theta" to distinguish from solar azimuth, which is measured clockwise from North.

    Returns:
        The origin of the wave front as a an array of shape (2,).
    """
    return (
        raster_n_rows
        * np.sin(_rad(theta))
        * np.array([np.sin(_rad(theta)), -np.cos(_rad(theta))])
    )


def _find_wave_front_width(
    wave_front_origin: tuple[float, float],
    theta: float,
    raster_n_cols: int,
) -> int:
    L1 = np.hypot(
        *wave_front_origin
    )  # distance from wave-front origin to raster origin
    L2 = (raster_n_cols) * np.cos(
        _rad(theta)
    )  # distance from raster origin to the upper-right corner of the wave front
    return L1 + L2


def _find_wave_front_front_length(
    wave_front_origin: tuple[float, float],
    raster_n_rows: int,
    raster_n_cols: int,
    theta: float,
) -> int:
    raster_bl_corner = np.array([raster_n_rows, 0])
    L3 = np.hypot(
        *(raster_bl_corner - wave_front_origin)
    )  # distance from wave-front origin to the bottom-left corner of the raster
    L4 = (
        (raster_n_cols) * np.sin(_rad(theta))
    )  # distance from the bottom-left corner of the raster to the bottom-left corner of the wave front
    return L3 + L4


def _make_wave_front(
    raster_n_rows: int,
    raster_n_cols: int,
    theta: float,
    packet_spacing: int = _packet_spacing,
    front_spacing: int = _front_spacing,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the discretized locations for a wave front that will cover a raster.
    For this purpose, locations are in pixel space.
    (0, 0) is the raster origin. (1, 1) is the bottom-right corner of the upper-left pixel.

    By "wave front", I just mean points on a grid that is at an angle theta from the the
    relevant raster grid. Each row of this grid can be thought of as a temporal snapshot of a wave front
    as it moves across the raster.

    Args:
        raster_n_rows: The number of rows in the raster.
        raster_n_cols: The number of columns in the raster.
        theta: (float in [0, 90]) The azimuth angle of the wave front in degrees counterclockwise from North.
               This is called "theta" to distinguish from solar azimuth, which is measured clockwise from North.
        packet_spacing: The spacing between points parallel to the front (spacing othorgonal to theta).
        front_spacing: The spacing between points orthogonal to the front (spacing along theta).

    Returns:
        Fi: The i-coordinates of the wave front according to the raster array axes.
            i.e. Fi[0, 0] is the horizontal distance between the wave-front and raster origins
            in units of pixels.
        Fj: The j-coordinates of the wave front according to the raster array axes.
    """
    if (theta < 0) or (theta > 90):
        raise ValueError(
            f"Theta {theta} must be between 0 and 90 degrees for make_wave_front."
        )
    wave_front_origin = _find_wave_front_origin(raster_n_rows, theta)
    n_packets = int(
        np.ceil(
            _find_wave_front_width(wave_front_origin, theta, raster_n_cols)
            / packet_spacing
        )
    )
    n_fronts = int(
        np.ceil(
            _find_wave_front_front_length(
                wave_front_origin, raster_n_rows, raster_n_cols, theta
            )
            / front_spacing
        )
    )

    hps = packet_spacing * np.cos(_rad(theta))  # horizontal packet spacing (in pixels)
    vps = packet_spacing * np.sin(_rad(theta))  # vertical packet spacing (in pixels)
    vfs = front_spacing * np.cos(_rad(theta))  # vertical front spacing (in pixels)
    hfs = front_spacing * np.sin(_rad(theta))  # horizontal front spacing (in pixels)

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


def _get_array_values_on_front(
    arr: np.ndarray,
    Fi: np.ndarray,
    Fj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the values of a raster on a wave front.
    Also returns intermediate data that are necessary to compute a shading mask.

    Args:
        arr: The array to get the values from.
        Fi: The i-coordinates of the wave front according to the raster array axes.
        Fj: The j-coordinates of the wave front according to the raster array axes.

    Returns:
        Fi_indices: The i-coordinates of the wave front rounded down to the nearest integer.
        Fj_indices: The j-coordinates of the wave front rounded down to the nearest integer.
        Fvalues: The values of the array on the wave front.
        valid_indices_on_front: The indices of the wave front that are inside the raster.
    """
    n_rows, n_cols = arr.shape
    Fi_indices = np.floor(Fi).astype(int)
    Fj_indices = np.floor(Fj).astype(int)
    valid_indices_on_front = (
        (Fi_indices >= 0)
        & (Fj_indices >= 0)
        & (Fi_indices < n_rows)
        & (Fj_indices < n_cols)
    )
    Fvalues = 0 * Fi - 9999
    Fvalues[valid_indices_on_front] = arr[
        Fi_indices[valid_indices_on_front], Fj_indices[valid_indices_on_front]
    ]
    return Fi_indices, Fj_indices, Fvalues, valid_indices_on_front


def _mean_over_indices(
    ii: np.ndarray, jj: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean of values for all unique pairs of indices in ii x jj.
    What this is for: You have a collection of values that are associated with
    points (i, j) where i and j are indices of a raster. But these points are not alligned
    with the raster grid. So you compute the mean of the values for all unique pairs of indices in ii x jj,
    and map those mean values to the raster grid.
    """
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
) -> np.ndarray:
    if elevation_angle <= 0:
        raise ValueError("Elevation angle must be greater than 0")
    if elevation_angle >= 90:
        raise ValueError("Elevation angle must be less than 90")
    grad_x = np.sin(_rad(azimuth_angle)) * np.tan(_rad(elevation_angle))
    grad_y = np.cos(_rad(azimuth_angle)) * np.tan(_rad(elevation_angle))
    return np.array([grad_x, grad_y])


def _add_gradient_to_dem(
    dem: Raster,
    grad_x: float,
    grad_y: float,
) -> Raster:
    return dem.with_array(
        dem.arr
        + np.vstack([np.arange(dem.arr.shape[1])] * dem.arr.shape[0]) * grad_x * dem.dx
        - np.vstack([np.arange(dem.arr.shape[0])] * dem.arr.shape[1]).transpose()
        * grad_y
        * dem.dy
    )


def _rotate_array(arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an array by an angle in degrees counterclockwise."""
    assert angle in [0, 90, 180, 270], ValueError(
        f"Angle {angle} must be 0, 90, 180, or 270 degrees"
    )
    if angle == 0:
        return arr
    elif angle == 90:
        return arr.T[::-1]
    elif angle == 180:
        return arr[::-1, ::-1]
    elif angle == 270:
        return arr.T[:, ::-1]
