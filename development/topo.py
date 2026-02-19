from pathlib import Path
import sys
import numpy as np

project_root = Path.cwd().parent  # Go up from dev directory
sys.path.insert(0, str(project_root))


from tests.conftest import make_flat_dem

n_rows, n_cols = 160, 120

dem = make_flat_dem(n_rows=n_rows, n_cols=n_cols)

# make a hill
y = np.arange(n_rows)
hill = 20 * np.exp(-((y - n_rows / 2) ** 2) / 10**2)
hill = hill[:, np.newaxis]
hill = np.tile(hill, (1, n_cols))

hill_dem = dem.with_array(hill)

# and some mounds
X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))


def make_mound(xc, yc, radius):
    return np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / radius**2)


mound_centers = [(n_cols / 2, n_rows / 4), (n_cols / 2, 3 * n_rows / 4)]
mound_heights = [25, 20]

mounds = sum(
    h * make_mound(xc, yc, 10) for h, (xc, yc) in zip(mound_heights, mound_centers)
)

mounds_dem = dem.with_array(mounds)

dem = dem.with_array(hill + mounds)
