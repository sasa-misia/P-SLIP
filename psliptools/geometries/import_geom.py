import shapely.geometry as geom
import shapely.ops as ops
import geopandas as gpd
import numpy as np

def load_shapefile_polygons_simple(
        shapefile_path: str, 
        field_name: str, 
        sel_filter: list = None
    ) -> tuple:
    """
    Load polygons from a shapefile, optionally filtering by field values.

    Args:
        shapefile_path (str): Path to the shapefile.
        field_name (str): Name of the field to filter on.
        selection (list, optional): List of values to select from field_name. If None, all features are loaded.

    Returns:
        tuple: (list of shapely.geometry.Polygon, list of selected field values)

    Raises:
        FileNotFoundError: If the shapefile does not exist.
        ValueError: If field_name is not present in the shapefile.
    """
    shape_gdf = gpd.read_file(shapefile_path)
    if sel_filter:
        sel_set = set(str(s) for s in sel_filter)
        shape_gdf = shape_gdf[shape_gdf[field_name].astype(str).isin(sel_set)]
    polygons = [geom.shape(geom_) if not isinstance(geom_, geom.Polygon) else geom_ for geom_ in shape_gdf.geometry]
    selected_names = shape_gdf[field_name].tolist()
    return polygons, selected_names

def load_shapefile_polygons(
        shapefile_path: str,
        field_name: str = None,
        poly_bound: geom.Polygon = None,
        mask_out_poly: bool = True,
        extra_bound: float = 0.0,
        sel_filter: list = None,
        points_lim: int = 80000,
        merge_by_class: bool = True
    ) -> tuple:
    """
    Load polygons from a shapefile with advanced options (filtering, bounding, masking, class selection).

    Args:
        shapefile_path (str): Path to the shapefile.
        field_name (str): Name of the field to group polygons, or 'None'.
        poly_bound (shapely.geometry.Polygon, optional): Polygon to use as bounding box (in lon/lat).
        mask_out_poly (bool, optional): If True, mask output polygons by poly_bound.
        extra_bound (float, optional): Extra boundary to apply (in degrees, not meters).
        sel_filter (list, optional): List of classes to select.
        points_lim (int, optional): Maximum number of points per polygon.
        merge_by_class (bool, optional): If True, merge polygons by class.

    Returns:
        tuple: (list of shapely.geometry.Polygon, list of selected field values)
    """
    shape_gdf = gpd.read_file(shapefile_path)
    # Ensure all geometries are shapely polygons
    if not shape_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError("All geometries in the shapefile must be Polygon or MultiPolygon types.")
    # shape_gdf['geometry'] = shape_gdf['geometry'].apply(lambda g: geom.shape(g) if not isinstance(g, geom.BaseGeometry) else g)

    # Filter by field/class if needed
    if field_name:
        if field_name not in shape_gdf.columns:
            raise ValueError(f"Field '{field_name}' not found in shapefile.")
        shape_gdf[field_name] = shape_gdf[field_name].astype(str) # Ensure field is string type for consistency
        if sel_filter:
            sel_filter = [str(s) for s in sel_filter]
            missing = [c for c in sel_filter if c not in set(shape_gdf[field_name])]
            if missing:
                raise ValueError(f"Some classes in sel_filter are not present: {missing}")
            shape_gdf = shape_gdf[shape_gdf[field_name].isin(sel_filter)]
            classes = sel_filter
        else:
            classes = sorted(shape_gdf[field_name].unique())
    else:
        field_name = 'None'
        classes = ['None']
        shape_gdf[field_name] = 'None'

    # Apply bounding polygon (with optional buffer)
    if poly_bound is not None and not poly_bound.is_empty:
        if extra_bound > 0:
            poly_bound = poly_bound.buffer(extra_bound)
        shape_gdf = shape_gdf[shape_gdf.geometry.intersects(poly_bound)]

    # Remove polygons with too many points
    def valid_geom(g):
        if isinstance(g, geom.MultiPolygon):
            total_points = sum(len(list(poly.exterior.coords)) for poly in g.geoms)
            return total_points <= points_lim
        elif hasattr(g, 'coords'):
            return len(list(g.coords)) <= points_lim
        else:
            raise ValueError("Geometry is not a valid Polygon or MultiPolygon.")
    shape_gdf = shape_gdf[shape_gdf.geometry.apply(valid_geom)]

    # Group and merge polygons by class if requested
    polygons = []
    selected_names = []
    for cls in classes:
        class_gdf = shape_gdf[shape_gdf[field_name] == cls]
        if class_gdf.empty:
            continue
        if merge_by_class:
            merged = ops.unary_union(class_gdf.geometry)
            geoms = merged.geoms if isinstance(merged, geom.MultiPolygon) else [merged]
        else:
            geoms = list(class_gdf.geometry)
        for poly in geoms:
            if poly.is_empty:
                continue
            polygons.append(poly)
            selected_names.append(cls)

    # Mask output polygons by poly_bound if requested
    if poly_bound is not None and mask_out_poly and not poly_bound.is_empty:
        masked_polygons = []
        masked_names = []
        for poly, name in zip(polygons, selected_names):
            inter = poly.intersection(poly_bound)
            if not inter.is_empty:
                if isinstance(inter, geom.MultiPolygon):
                    for p in inter.geoms:
                        masked_polygons.append(p)
                        masked_names.append(name)
                else:
                    masked_polygons.append(inter)
                    masked_names.append(name)
        polygons = masked_polygons
        selected_names = masked_names

    if len(polygons) != len(selected_names):
        raise RuntimeError("Output polygons and names have different lengths.")

    return polygons, selected_names