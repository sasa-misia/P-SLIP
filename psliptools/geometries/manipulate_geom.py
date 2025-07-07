import shapely.geometry as geom

def intersect_polygons(polygons: geom.Polygon, mask: geom.Polygon) -> list:
    """Intersect a list of polygons with a mask polygon (or MultiPolygon)."""
    return [p.intersection(mask) for p in polygons if not p.is_empty and not p.intersection(mask).is_empty]