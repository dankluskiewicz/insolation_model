"""This is just the code I used to download data for a test DEM."""

import elevation
from context import dem_path


bounds = (-121.6, 47.8, -121.4, 47.9)
elevation.clip(bounds=bounds, output=str(dem_path), product="SRTM1")
