import pandas as pd
import shapely.geometry as geom
import shapely.ops as ops

def intersect_polygons(polygons: list, mask: geom.Polygon) -> list:
    """Intersect a list of polygons with a mask polygon (or MultiPolygon)."""
    if isinstance(polygons, pd.Series):
        polygons = polygons.tolist()
    
    intersected_poly = [
        p.intersection(mask) 
        for p in polygons 
        if not p.is_empty and not p.intersection(mask).is_empty
    ]
    return intersected_poly

def union_polygons(polygons: list) -> geom.Polygon:
    """Union a list of polygons (or MultiPolygon) into a single polygon (or MultiPolygon)."""
    if isinstance(polygons, pd.Series):
        polygons = polygons.tolist()
    if not polygons:
        raise ValueError("No polygons provided for union.")
    for p in polygons:
        if not isinstance(p, (geom.Polygon, geom.MultiPolygon)):
            raise ValueError("All elements must be shapely Polygon or MultiPolygon objects.")
    return ops.unary_union(polygons)