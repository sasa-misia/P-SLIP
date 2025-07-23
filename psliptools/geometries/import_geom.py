#%% # Import necessary libraries
import shapely.geometry as geom
import shapely.ops as ops
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

#%% # Function to check and fix geometries in a GeoDataFrame
def check_and_fix_geometries(shape_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Check and fix invalid geometries in a GeoDataFrame.
    If any geometry is invalid, it will be fixed using buffer(0).
    Issues a warning if any geometry is fixed.
    
    Args:
        shape_gdf (gpd.GeoDataFrame): The GeoDataFrame to check.
        
    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with fixed geometries.
    """
    out_shape_gdf = shape_gdf.copy()
    invalid_mask = ~out_shape_gdf.geometry.is_valid
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        warnings.warn(f"{n_invalid} invalid geometries found and fixed using buffer(0).", UserWarning)
        out_shape_gdf.loc[invalid_mask, 'geometry'] = out_shape_gdf.loc[invalid_mask, 'geometry'].buffer(0)
    return out_shape_gdf

#%% # Function to convert a GeoDataFrame to EPSG:4326 (WGS84)
def convert_gdf_to_geo(shape_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert a GeoDataFrame to EPSG:4326 (WGS84) if it is not already in that CRS.

    Args:
        shape_gdf (gpd.GeoDataFrame): The GeoDataFrame to convert.

    Returns:
        gpd.GeoDataFrame: The converted GeoDataFrame in EPSG:4326.
    """
    if shape_gdf.crs is not None and shape_gdf.crs.to_epsg() != 4326:
        return check_and_fix_geometries(shape_gdf.to_crs(epsg=4326))
    return check_and_fix_geometries(shape_gdf)

#%% # Function to validate geometry in a GeoDataFrame row
def valid_geom_row(row: pd.Series, points_lim: int = 80000) -> bool:
    """
    Check if the geometry in a GeoDataFrame row is valid and does not exceed the points limit.
    Issues a warning with the row index if the geometry is invalid or too complex.

    Args:
        row (pd.Series): A row from a GeoDataFrame.
        points_lim (int): Maximum allowed number of points.

    Returns:
        bool: True if geometry is valid and within the limit, False otherwise.
    """
    row_geom = row.geometry
    idx = row.name
    if row_geom.is_empty:
        warnings.warn(f"Row {idx} geometry is empty.", UserWarning)
        return False
    elif isinstance(row_geom, geom.MultiPolygon):
        points_tot = sum(len(list(poly.exterior.coords)) for poly in row_geom.geoms)
        if points_tot > points_lim:
            warnings.warn(f"Row {idx} Polygon exceeds points limit ({points_tot} > {points_lim})", UserWarning)
            return False
    elif isinstance(row_geom, geom.Polygon):
        points_tot = len(list(row_geom.exterior.coords))
        if points_tot > points_lim:
            warnings.warn(f"Row {idx} Polygon exceeds points limit ({points_tot} > {points_lim})", UserWarning)
            return False
    else:
        warnings.warn(f"Row {idx} geometry is not a valid Polygon or MultiPolygon.", UserWarning)
        return False
    return True

#%% # Function to check if a polygon is valid and in EPSG:4326
def check_poly_bound(poly_bound: geom.base.BaseGeometry) -> None:
    """
    Check that poly_bound is a Polygon or MultiPolygon in EPSG:4326 (lat/lon) and coordinates are in valid ranges.
    Raises ValueError if type is not Polygon/MultiPolygon, or if coordinates are outside typical lon/lat ranges.

    Args:
        poly_bound (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The polygon or multipolygon to check.

    Raises:
        ValueError: If poly_bound is not a Polygon/MultiPolygon, or not in EPSG:4326, or coordinates are out of range.
    """
    # Check type
    if not isinstance(poly_bound, (geom.Polygon, geom.MultiPolygon)):
        raise ValueError("poly_bound must be a shapely Polygon or MultiPolygon.")

    # Shapely geometries do not have a 'crs' attribute!
    # If poly_bound is a GeoSeries or GeoDataFrame, user should pass .unary_union or .geometry[0] instead.

    # Check coordinates range
    coords = []
    if isinstance(poly_bound, geom.Polygon):
        coords = list(poly_bound.exterior.coords)
    elif isinstance(poly_bound, geom.MultiPolygon):
        for p in poly_bound.geoms:
            coords.extend(list(p.exterior.coords))
    if coords:
        lons, lats = zip(*coords)
        if not (all(-180 <= x <= 180 for x in lons) and all(-90 <= y <= 90 for y in lats)):
            raise ValueError("poly_bound coordinates are outside typical lon/lat ranges. Ensure it is in EPSG:4326.")
        
#%% # Function to load polygons from a shapefile with optional filtering
def load_shapefile_polygons_simple(
        shapefile_path: str, 
        field_name: str, 
        sel_filter: list = None
    ) -> pd.DataFrame:
    """
    Load polygons from a shapefile, optionally filtering by field values.

    Args:
        shapefile_path (str): Path to the shapefile.
        field_name (str): Name of the field to filter on.
        selection (list, optional): List of values to select from field_name. If None, all features are loaded.

    Returns:
        pd.DataFrame: DataFrame with columns 'class_name' and 'geometry'

    Raises:
        FileNotFoundError: If the shapefile does not exist.
        ValueError: If field_name is not present in the shapefile.
    """
    shape_gdf = convert_gdf_to_geo(gpd.read_file(shapefile_path))
    if sel_filter:
        sel_set = set(str(s) for s in sel_filter)
        shape_gdf = shape_gdf[shape_gdf[field_name].astype(str).isin(sel_set)]
    # Ensure all geometries are shapely polygons
    sel_polys = [
        geom.shape(g) if not isinstance(g, (geom.Polygon, geom.MultiPolygon)) 
        else g
        for g in shape_gdf.geometry
    ]
    sel_names = [str(val) for val in shape_gdf[field_name].tolist()] # Ensure names are strings
    return pd.DataFrame({'class_name': sel_names, 'geometry': sel_polys})

#%% # Function to load polygons from a shapefile with advanced options
def load_shapefile_polygons(
        shapefile_path: str,
        field_name: str = None,
        sel_filter: list = None,
        poly_bound: geom.base.BaseGeometry = None,  # Polygon or MultiPolygon
        mask_out_poly: bool = True,
        extra_bound: float = 0.0,
        points_lim: int = 80000,
    ) -> pd.DataFrame:
    """
    Load polygons from a shapefile with advanced options (filtering, bounding, masking, class selection).

    Args:
        shapefile_path (str): Path to the shapefile.
        field_name (str, optional): Name of the field to group polygons, or 'None'.
        sel_filter (list, optional): List of classes to select.
        poly_bound (shapely.geometry.Polygon or shapely.geometry.MultiPolygon, optional): Polygon or MultiPolygon to use as bounding box (in lon/lat, EPSG:4326).
        mask_out_poly (bool, optional): If True, mask output polygons by poly_bound.
        extra_bound (float, optional): Extra boundary to apply (in degrees, not meters).
        points_lim (int, optional): Maximum number of points per polygon.

    Returns:
        pd.DataFrame: DataFrame with columns 'class_name' and 'geometry'.
    """
    shape_gdf = convert_gdf_to_geo(gpd.read_file(shapefile_path))
    # Ensure all geometries are shapely polygons
    if not shape_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError("All geometries in the shapefile must be Polygon or MultiPolygon types.")

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
            sel_classes = sel_filter
        else:
            sel_classes = sorted(shape_gdf[field_name].unique())
    else:
        # Create a new field explicitly and merge all polygons
        field_name = 'ALL_POLYGONS_MERGED'
        shape_gdf[field_name] = 'ALL_POLYGONS_MERGED' # String type for consistency
        sel_classes = ['ALL_POLYGONS_MERGED']

    # Check poly_bound CRS (must be in lat/lon)
    if poly_bound is not None and not poly_bound.is_empty:
        check_poly_bound(poly_bound)
        if extra_bound > 0:
            poly_bound = poly_bound.buffer(extra_bound)
        shape_gdf = shape_gdf[shape_gdf.geometry.intersects(poly_bound)] # This is to filter geometries that intersect the bounding polygon, without clipping them

    # Remove polygons with too many points
    shape_gdf = shape_gdf[shape_gdf.apply(lambda row: valid_geom_row(row, points_lim=points_lim), axis=1)]

    # Group and merge polygons by class
    sel_polys = []
    sel_names = []
    for cls in sel_classes:
        sel_cls_gdf = shape_gdf[shape_gdf[field_name] == cls]
        if sel_cls_gdf.empty:
            continue
        # Always merge all geometries for this class into a single geometry (Polygon or MultiPolygon)
        merged_poly = ops.unary_union(sel_cls_gdf.geometry)
        # If the result is a MultiPolygon or Polygon, just append it as a single object
        if not merged_poly.is_empty:
            sel_polys.append(merged_poly)
            sel_names.append(cls)

    # Mask output polygons by poly_bound if requested
    if poly_bound is not None and mask_out_poly and not poly_bound.is_empty:
        masked_polys = []
        masked_names = []
        for poly, name in zip(sel_polys, sel_names):
            # Support both Polygon and MultiPolygon for poly_bound
            if isinstance(poly_bound, geom.MultiPolygon):
                inter = [poly.intersection(pb) for pb in poly_bound.geoms]
                inter = [g for g in inter if not g.is_empty]
                if inter:
                    merged = ops.unary_union(inter)
                    if not merged.is_empty:
                        masked_polys.append(merged)
                        masked_names.append(name)
            else:
                inter = poly.intersection(poly_bound)
                if not inter.is_empty:
                    masked_polys.append(inter)
                    masked_names.append(name)
        sel_polys = masked_polys
        sel_names = masked_names

    if len(sel_polys) != len(sel_names):
        raise RuntimeError("Output polygons and their names have different lengths.")

    # Output always as DataFrame
    return pd.DataFrame({'class_name': sel_names, 'geometry': sel_polys})