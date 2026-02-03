from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from insolation_model.raster import Raster
from insolation_model.geometry.topography import dem_to_hillshade


def raster(
    raster: Raster,
    ax: Axes | None = None,
    cbar: bool = False,
    **kwargs,
) -> Axes:
    """Display a raster as an image.

    Args:
        raster: Raster to display.
        ax: Matplotlib axes. If None, uses current axes.
        cbar: Whether to show a colorbar.
        **kwargs: Passed to ax.imshow (e.g. cmap, vmin, vmax).
    """
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


def hillshade(
    dem: Raster,
    ax: Axes | None = None,
    solar_azimuth: float = 315,
    solar_elevation: float = 45,
    cbar: bool = False,
    **kwargs,
) -> Axes:
    """Display a DEM as hillshade (shaded relief).

    Args:
        dem: Digital elevation model raster to display as hillshade for.
        ax: Matplotlib axes. If None, uses current axes.
        solar_azimuth: Light direction in degrees clockwise from north. Default 315 (NW).
        solar_elevation: Light angle in degrees above horizon. Default 45.
        cbar: Whether to show a colorbar.
        **kwargs: Passed to ax.imshow (e.g. cmap, vmin, vmax).
    """
    ax = _get_ax(ax)
    hill_shade = dem_to_hillshade(
        dem, solar_azimuth=solar_azimuth, solar_elevation=solar_elevation
    )

    imshow_kwargs = {"cmap": "viridis", "vmin": -1, "vmax": 1}
    imshow_kwargs.update(kwargs)
    return raster(hill_shade, ax=ax, cbar=cbar, **imshow_kwargs)


def _get_ax(ax: Axes | None) -> Axes:
    return ax or plt.gca()
