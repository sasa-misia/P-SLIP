"""
Geometry utilities for the psliptools package.

Provides functions for importing and generating geometric objects (e.g., polygons).
"""

from .generate_geom import (
    get_rectangle_parameters,
    create_rectangle_polygons
)

from .import_geom import (
    convert_gdf_to_geo,
    load_shapefile_polygons_simple,
    load_shapefile_polygons
)

from .manipulate_geom import (
    intersect_polygons,
    union_polygons
)

from .info_geom import (
    get_polygon_extremes,
    get_shapefile_fields,
    get_shapefile_field_values
)

__all__ = [
    "get_rectangle_parameters",
    "create_rectangle_polygons",
    "convert_gdf_to_geo",
    "load_shapefile_polygons_simple",
    "load_shapefile_polygons",
    "intersect_polygons",
    "union_polygons",
    "get_polygon_extremes",
    "get_shapefile_fields",
    "get_shapefile_field_values"
]