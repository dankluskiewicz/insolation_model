import pytest
import numpy as np

from insolation_model.geometry.shading import (
    _rotate_points_around_z_axis,
)

# fmt: off
N = np.array([[0,], [1,], [0,]])
E = np.array([[1,], [0,], [0,]])
S = np.array([[0,], [-1,], [0,]])
W = np.array([[-1,], [0,], [0,]])
U = np.array([[0,], [0,], [1,]])
D = np.array([[0,], [0,], [-1,]])
# fmt: on

NE = (N + E) / np.linalg.norm(N + E)
NW = (N + W) / np.linalg.norm(N + W)
SE = (S + E) / np.linalg.norm(S + E)
SW = (S + W) / np.linalg.norm(S + W)


@pytest.mark.parametrize("vector", [N, E, S, W, U, D])
@pytest.mark.parametrize("angle", [0, 15, 37, 90, 180, 200, 270, 360, 400])
def test_rotation_preserves_length(vector, angle):
    rotated = _rotate_points_around_z_axis(vector, angle)
    assert np.isclose(np.linalg.norm(rotated), 1.0, atol=1e-5)


@pytest.mark.parametrize(
    ["input_vector", "output_vector", "angle"],
    [
        (N, W, 90),
        (E, N, 90),
        (S, E, 90),
        (W, S, 90),
        (U, U, 90),
        (D, D, 90),
        (N, S, 180),
        (E, W, 180),
        (D, D, 180),
        (NE, SW, 180),
        (NW, SE, 180),
        (NE, NE, 720),
        (NW, N, -45),
        (
            np.concatenate([N, E, S, W, U, D], axis=1),
            np.concatenate([W, N, E, S, U, D], axis=1),
            90,
        ),
        (
            np.concatenate([N, E, S, W, U, D], axis=1),
            np.concatenate([E, S, W, N, U, D], axis=1),
            270,
        ),
    ],
)
def test_z_axis_rotation_converts_directions_correctly(
    input_vector, output_vector, angle
):
    rotated = _rotate_points_around_z_axis(input_vector, angle)
    assert np.isclose(rotated, output_vector).all(), (
        f"{input_vector=}, {np.round(rotated, 1)=}, {output_vector=}, {angle=}"
    )
