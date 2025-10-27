"""
Raster utilities for the psliptools package.

Provides functions for importing and generating raster grids (e.g., elevation).
"""

from .coordinates import (
    get_georaster_info,
    get_xy_grids_from_profile,
    get_bbox_from_profile,
    get_projected_epsg_code_from_bbox,
    get_unit_of_measure_from_epsg,
    get_projected_crs_from_bbox,
    are_coords_geographic,
    create_bbox_from_grids,
    create_grid_from_bbox,
    convert_coords,
    convert_coords_from_list,
    convert_coords_to_geo,
    transformer_from_grids,
    convert_grids_and_profile_to_geo,
    convert_bbox,
    get_pixels_inside_polygon,
    raster_within_polygon,
    get_closest_pixel_idx
)

from .manage_raster import (
    get_1d_idx_from_2d_mask,
    get_1d_idx_from_2d_idx,
    get_2d_idx_from_1d_idx,
    get_2d_mask_from_1d_idx,
    mask_raster_with_1d_idx,
    pick_point_from_1d_idx
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

from .generate_raster import (
    generate_grids_from_indices,
    generate_slope_and_aspect_rasters,
    generate_curvature_rasters
)

__all__ = [
    "load_georaster",
    "get_georaster_info",
    "get_xy_grids_from_profile",
    "get_bbox_from_profile",
    "get_projected_epsg_code_from_bbox",
    "get_unit_of_measure_from_epsg",
    "get_projected_crs_from_bbox",
    "are_coords_geographic",
    "get_pixels_inside_polygon",
    "raster_within_polygon",
    "get_closest_pixel_idx",
    "get_1d_idx_from_2d_mask",
    "get_1d_idx_from_2d_idx",
    "get_2d_idx_from_1d_idx",
    "get_2d_mask_from_1d_idx",
    "mask_raster_with_1d_idx",
    "pick_point_from_1d_idx",
    "create_bbox_from_grids",
    "create_grid_from_bbox",
    "convert_coords",
    "convert_coords_from_list",
    "convert_coords_to_geo",
    "transformer_from_grids",
    "convert_grids_and_profile_to_geo",
    "convert_bbox",
    "replace_values",
    "resample_raster",
    "plot_elevation_2d",
    "plot_elevation_isometric",
    "plot_elevation_3d",
    "generate_grids_from_indices",
    "generate_slope_and_aspect_rasters",
    "generate_curvature_rasters"
]