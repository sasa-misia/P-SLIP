import shapely.geometry as geom

def get_polygon_extremes(polygon: geom.Polygon) -> tuple:
    """Get min/max coordinates from a polygon."""
    if hasattr(polygon, "exterior"):
        xs, ys = polygon.exterior.xy
        return (min(xs), min(ys)), (max(xs), max(ys))
    return (None, None), (None, None)