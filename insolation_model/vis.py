from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from insolation_model.raster import Raster


def _get_ax(ax: Axes | None) -> Axes:
    return ax or plt.gca()


def raster(
    raster: Raster,
    ax: Axes | None = None,
    cbar: bool = False,
    **kwargs,
) -> Axes:
    ax = _get_ax(ax)
    xmin, ymin, xmax, ymax = raster.bounds

    image = ax.imshow(
        raster.arr,
        extent=(xmin, xmax, ymin, ymax),
        **kwargs,
    )
    if cbar and ax.figure is not None:
        ax.figure.colorbar(image, ax=ax, orientation="vertical", fraction=0.1)
    return ax
