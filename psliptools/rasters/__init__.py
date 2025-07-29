"""
Raster utilities for the psliptools package.

Provides functions for importing and generating raster grids (e.g., elevation).
"""

from .coordinates import (
    get_georaster_info,
    get_xy_grids_from_profile,
    get_projected_epsg_code_from_bbox,
    get_projected_crs_from_bbox,
    is_geographic_coords,
    create_bbox,
    create_grid_from_bbox,
    convert_coords,
    convert_coords_to_geo,
    transformer_from_grids,
    convert_grids_and_profile_to_geo,
    convert_bbox,
    get_pixels_inside_polygon,
    raster_within_polygon
)

from .manage_raster import (
    get_2d_idx_from_1d_idx,
    get_2d_mask_from_1d_idx,
    mask_raster_with_1d_idx
)

from .manipulate_raster import (
    replace_values,
    resample_raster
)

from .plot_raster import (
    plot_elevation_2d,
    plot_elevation_isometric,
    plot_elevation_3d
)

from .import_raster import (
    load_georaster
)

__all__ = [
    "load_georaster",
    "get_georaster_info",
    "get_xy_grids_from_profile",
    "get_projected_epsg_code_from_bbox",
    "get_projected_crs_from_bbox",
    "is_geographic_coords",
    "get_pixels_inside_polygon",
    "raster_within_polygon",
    "get_2d_idx_from_1d_idx",
    "get_2d_mask_from_1d_idx",
    "mask_raster_with_1d_idx",
    "create_bbox",
    "create_grid_from_bbox",
    "convert_coords",
    "convert_coords_to_geo",
    "transformer_from_grids",
    "convert_grids_and_profile_to_geo",
    "convert_bbox",
    "replace_values",
    "resample_raster",
    "plot_elevation_2d",
    "plot_elevation_isometric",
    "plot_elevation_3d"
]