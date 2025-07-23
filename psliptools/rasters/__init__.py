"""
Raster utilities for the psliptools package.

Provides functions for importing and generating raster grids (e.g., elevation).
"""

from .import_raster import (
    load_georaster
)

from .info_raster import (
    get_georaster_info,
    get_xy_grids_from_profile,
    get_projected_epsg_code_from_bbox,
    get_projected_crs_from_bbox
)

from .manipulate_raster import (
    create_bbox,
    create_grid_from_bbox,
    convert_coords,
    convert_coords_to_geo,
    transformer_from_grids,
    convert_grids_and_profile_to_geo,
    convert_bbox,
    replace_values,
    resample_raster
)

from .show_raster import (
    show_elevation_isometric,
    show_elevation_3d
)

__all__ = [
    "load_georaster",
    "get_georaster_info",
    "get_xy_grids_from_profile",
    "get_projected_epsg_code_from_bbox",
    "get_projected_crs_from_bbox",
    "create_bbox",
    "create_grid_from_bbox",
    "convert_coords",
    "convert_coords_to_geo",
    "transformer_from_grids",
    "convert_grids_and_profile_to_geo",
    "convert_bbox",
    "replace_values",
    "resample_raster",
    "show_elevation_isometric",
    "show_elevation_3d"
]