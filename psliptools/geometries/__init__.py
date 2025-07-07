"""
Geometry utilities for the psliptools package.

Provides functions for importing and generating geometric objects (e.g., polygons).
"""

from .generate_geom import (
    create_rectangle_polygons
)

from .import_geom import (
    load_shapefile_polygons
)

from .manipulate_geom import (
    intersect_polygons
)

from .info_geom import (
    get_polygon_extremes
)

__all__ = [
    "create_rectangle_polygons",
    "load_shapefile_polygons",
    "intersect_polygons",
    "get_polygon_extremes"
]