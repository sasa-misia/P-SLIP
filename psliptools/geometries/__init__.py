"""
Geometry utilities for the psliptools package.

Provides functions for importing and generating geometric objects (e.g., polygons).
"""

from .generate_geom import (
    get_rectangle_parameters,
    create_rectangle_polygons
)

from .manipulate_geom import (
    intersect_polygons,
    union_polygons,
    subtract_polygons,
    get_ext_int_coords_from_polygon,
    create_polygon_from_coord_lists,
    convert_simple_polygon_crs,
    convert_polygons_crs,
    add_buffer_to_polygons
)

from .info_geom import (
    get_polygon_extremes,
    get_shapefile_fields,
    get_shapefile_field_values
)

from .import_geom import (
    convert_gdf_to_geo,
    load_shapefile_polygons_simple,
    load_shapefile_polygons
)

__all__ = [
    "get_rectangle_parameters",
    "create_rectangle_polygons",
    "convert_gdf_to_geo",
    "load_shapefile_polygons_simple",
    "load_shapefile_polygons",
    "intersect_polygons",
    "union_polygons",
    "subtract_polygons",
    "get_ext_int_coords_from_polygon",
    "create_polygon_from_coord_lists",
    "convert_simple_polygon_crs",
    "convert_polygons_crs",
    "add_buffer_to_polygons",
    "get_polygon_extremes",
    "get_shapefile_fields",
    "get_shapefile_field_values"
]