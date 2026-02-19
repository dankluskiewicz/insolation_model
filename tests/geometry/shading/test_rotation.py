import numpy as np
from insolation_model.geometry.shading import _rotate_array


def test_rotate_array():
    arr = np.zeros((4, 4))
    arr[0, 0] = 1
    for angle, new_ij in [(0, (0, 0)), (90, (0, 3)), (180, (3, 3)), (270, (3, 0))]:
        rotated_arr = _rotate_array(arr, angle)
        assert rotated_arr[new_ij] == 1
        assert rotated_arr.flatten().sum() == 1
