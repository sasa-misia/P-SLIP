# %% === Import necessary modules
import pandas as pd
import shapely.geometry as geom
import shapely.ops as ops
import warnings

from psliptools.rasters.coordinates import convert_coords_from_list, are_coords_geographic, get_projected_epsg_code_from_bbox

# %% === Function to check the geometry of a polygon
def _check_polygon_is_valid(
    polygon: geom.Polygon | geom.MultiPolygon
    ) -> bool:
    if not isinstance(polygon, (geom.Polygon, geom.MultiPolygon)):
        raise ValueError("Input must be a shapely Polygon or MultiPolygon.")
    is_valid = polygon.is_valid
    if not isinstance(is_valid, bool):
        raise ValueError("Internal error while checking polygon geometry.")
    return is_valid

# %% === Function to check the geometry of a polygon and fix it if necessary
def _check_and_fix_polygon(
        polygon: geom.Polygon | geom.MultiPolygon
    ) -> None:
    if not isinstance(polygon, (geom.Polygon, geom.MultiPolygon)):
        raise ValueError("Input must be a shapely Polygon or MultiPolygon.")
    if not _check_polygon_is_valid(polygon):
        polygon = polygon.buffer(0)
        if not _check_polygon_is_valid(polygon):
            raise ValueError("Tried to fix polygon geometry, but it is still invalid.")
    return polygon

# %% === Function to check content of a list of supposed polygons (or MultiPolygon)
def _check_and_collect_polygons_in_list(
        polygons: geom.Polygon | geom.MultiPolygon | list | pd.Series
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Check that a list of polygons (or MultiPolygon) is valid and if not, convert it.

    Args:
        polygons (list | pd.Series): List of polygons (or MultiPolygon) to check.
    
    Returns:
        list: List of polygons (or MultiPolygon).

    Raises:
        ValueError: If a polygon is not a Polygon or MultiPolygon.
    """
    if isinstance(polygons, pd.Series):
        polygons = polygons.tolist()
    if not polygons:
        raise ValueError("No polygons provided for union.")
    if not isinstance(polygons, list):
        polygons = [polygons]
    for idx, p in enumerate(polygons):
        if not isinstance(p, (geom.Polygon, geom.MultiPolygon)):
            raise ValueError(f"Polygon at index {idx} is not a Polygon or MultiPolygon.")
    return polygons

# %% === Function to create a single polygon from a list of polygons
def union_polygons(
        polygons: list | pd.Series
    ) -> geom.Polygon | geom.MultiPolygon:
    """
    Union a list of polygons (or MultiPolygon) into a single polygon (or MultiPolygon).

    Args:
        polygons (list | pd.Series): List of polygons (or MultiPolygon) to union.

    Returns:
        geom.Polygon | geom.MultiPolygon: Unioned polygon (or MultiPolygon).
    """
    polygons = _check_and_collect_polygons_in_list(polygons)
    unioned_polygon = ops.unary_union(polygons)
    return unioned_polygon

# %% === Function to intersect a list of polygons with a mask polygon
def intersect_polygons(
        polygons: geom.Polygon | geom.MultiPolygon | list | pd.Series, 
        mask: geom.Polygon | geom.MultiPolygon | list | pd.Series,
        clean_empty: bool = False
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Intersect a list of polygons with a mask polygon (or MultiPolygon).

    Args:
        polygons (geom.Polygon | geom.MultiPolygon | list | pd.Series): Polygons (or MultiPolygon) to intersect.
        mask (geom.Polygon | geom.MultiPolygon | list | pd.Series): Mask polygon (or MultiPolygon) to intersect with.
        clean_empty (bool, optional): Whether to remove empty polygons from the result. Defaults to False.

    Returns:
        list: List of intersected polygons (or MultiPolygon).
    """
    polygons = _check_and_collect_polygons_in_list(polygons)
    mask_union = union_polygons(mask)

    intersected_poly = []
    for p in polygons:
        if isinstance(p, geom.MultiPolygon):
            p_inter_multi = [pp.intersection(mask_union) for pp in p.geoms]
            p_inter_multi = [pp for pp in p_inter_multi if not pp.is_empty]
            p_intersection = union_polygons(p_inter_multi)
        elif isinstance(p, geom.Polygon):
            p_intersection = p.intersection(mask_union)
        else:
            raise ValueError("each element of polygons must be a shapely Polygon or MultiPolygon.")
        
        if p_intersection.is_empty:
            intersected_poly.append(None)
        else:
            intersected_poly.append(p_intersection)
    
    # intersected_poly = [
    #     p.intersection(mask_union)
    #     if not p.is_empty and not p.intersection(mask_union).is_empty
    #     else None
    #     for p in polygons
    # ]

    if clean_empty:
        intersected_poly = [p for p in intersected_poly if p]

    if not intersected_poly:
        warnings.warn("The intersection of the polygons with the mask is empty.")
    return intersected_poly

# %% === Function to subtract a polygon from a list of polygons
def subtract_polygons(
        polygons: geom.Polygon | geom.MultiPolygon | list | pd.Series, 
        subtract_mask: geom.Polygon | geom.MultiPolygon | list | pd.Series,
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Subtract a list of polygons (or MultiPolygon) from another list of polygons (or MultiPolygon).

    Args:
        polygons (geom.Polygon | geom.MultiPolygon | list | pd.Series): Polygons (or MultiPolygon) to be subtracted.
        subtract (geom.Polygon | geom.MultiPolygon | list | pd.Series): Mask polygon (or MultiPolygon) to subtract.

    Returns:
        list: List of subtracted polygons (or MultiPolygon).
    """
    polygons = _check_and_collect_polygons_in_list(polygons)
    subtract_union = union_polygons(subtract_mask)

    subtracted_polygon = [p.symmetric_difference(subtract_union) for p in polygons]
    return subtracted_polygon

# %% === Function to obtain list of coordinates (exterior and interiors) from a polygon
def get_ext_int_coords_from_polygon(
        polygon: geom.Polygon
    ) -> tuple[list, list]:
    """
    Obtain list of coordinates from a polygon

    Args:
        polygon (geom.Polygon): Polygon to obtain coordinates from.

    Returns:
        tuple[list, list]: List of x and y coordinates (first element inside each list is the exterior and the rest are the interiors).
    """
    if not isinstance(polygon, geom.Polygon):
        raise ValueError("Input must be a shapely Polygon, not MultiPolygon or something else.")
    x_coords_poly_list, y_coords_poly_list = [], []
    x_coords_poly_list.append(polygon.exterior.coords.xy[0])
    y_coords_poly_list.append(polygon.exterior.coords.xy[1])
    for interior in polygon.interiors:
        x_coords_poly_list.append(interior.coords.xy[0])
        y_coords_poly_list.append(interior.coords.xy[1])
    return x_coords_poly_list, y_coords_poly_list

# %% === Function to create a polygon from a list of coordinates
def create_polygon_from_coord_lists(
        x_coords: list,
        y_coords: list
    ) -> geom.Polygon:
    """
    Create a polygon from a list of x and y coordinates.

    Args:
        x_coords (list): List of x coordinates, where each element of the list contains an array of x coordinates (first element inside the list is the exterior and the rest are the interiors).
        y_coords (list): List of y coordinates, where each element of the list contains an array of y coordinates (first element inside the list is the exterior and the rest are the interiors).

    Returns:
        geom.Polygon: Polygon created from x and y coordinates.
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")
    for idx, (x_points, y_points) in enumerate(zip(x_coords, y_coords)):
        if len(x_points) != len(y_points):
            raise ValueError(f"x_coords and y_coords at index {idx} must have the same length.")
    exterior = [(x, y) for x, y in zip(x_coords[0], y_coords[0])]
    if len(x_coords) > 1:
        interiors = []
        for x_points, y_points in zip(x_coords[1:], y_coords[1:]):
            interiors.append([(x, y) for x, y in zip(x_points, y_points)])
        out_polygon = geom.Polygon(exterior, holes=interiors)
    else:
        out_polygon = geom.Polygon(exterior)
    return out_polygon

# %% === Function to convert a single polygon from one CRS to another
def convert_simple_polygon_crs(
    polygon: geom.Polygon,
    crs_in: int,
    crs_out: int
    ) -> geom.Polygon:
    """
    Convert coordinates of a polygon from one CRS to another.

    Args:
        polygons (geom.Polygon): Polygon to convert (not MultiPolygon).
        crs_in (int): The EPSG code of the input coordinate reference system.
        crs_out (int): The EPSG code of the output coordinate reference system.

    Returns:
        geom.Polygon: Converted polygon.
    """
    if not isinstance(polygon, geom.Polygon):
        raise ValueError("Input must be a shapely Polygon, not MultiPolygon or something else.")
    x_coords_poly_in, y_coords_poly_in = get_ext_int_coords_from_polygon(polygon)
    x_coords_poly_out, y_coords_poly_out = convert_coords_from_list(crs_in, crs_out, x_coords_poly_in, y_coords_poly_in)
    polygon_out = create_polygon_from_coord_lists(x_coords_poly_out, y_coords_poly_out)
    return polygon_out

# %% === Function to convert multiple polygons from one CRS to another
def convert_polygons_crs(
        polygons: geom.Polygon | geom.MultiPolygon | list | pd.Series,
        crs_in: int | list[int],
        crs_out: int | list[int]
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Convert coordinates of a polygon (or MultiPolygon, or list of polygons) from one CRS to another.

    Args:
        polygons (geom.Polygon | geom.MultiPolygon | list | pd.Series): Polygons (or MultiPolygon) to convert.
        crs_in (int): The EPSG code of the input coordinate reference system.
        crs_out (int): The EPSG code of the output coordinate reference system.

    Returns:
        list: List of converted polygons (or MultiPolygons).
    """
    polygons = _check_and_collect_polygons_in_list(polygons)

    if isinstance(crs_in, int):
        crs_in_list = [crs_in for _ in range(len(polygons))]
    elif isinstance(crs_in, list):
        if len(crs_in) != len(polygons):
            raise ValueError("crs_in and polygons must have the same length.")
        crs_in_list = crs_in
    else:
        raise ValueError("crs_in must be an int or a list of ints.")
    
    if isinstance(crs_out, int):
        crs_out_list = [crs_out for _ in range(len(polygons))]
    elif isinstance(crs_out, list):
        if len(crs_out) != len(polygons):
            raise ValueError("crs_out and polygons must have the same length.")
        crs_out_list = crs_out
    else:
        raise ValueError("crs_out must be an int or a list of ints.")
    
    polygons_out = polygons.copy()
    for idx, p in enumerate(polygons):
        if isinstance(p, geom.Polygon):
            polygons_out[idx] = _check_and_fix_polygon(convert_simple_polygon_crs(p, crs_in_list[idx], crs_out_list[idx]))
        elif isinstance(p, geom.MultiPolygon):
            temp_poly_list = []
            for pp in p.geoms:
                temp_poly_list.append(
                    _check_and_fix_polygon(convert_simple_polygon_crs(pp, crs_in_list[idx], crs_out_list[idx]))
                )
            polygons_out[idx] = geom.MultiPolygon(temp_poly_list)
        else:
            raise ValueError("each element of polygons must be a shapely Polygon or MultiPolygon.")
    return polygons_out

# %% ===  Function to add a buffer (in meters) to a polygon
def add_buffer_to_polygons(
        polygons: geom.Polygon | geom.MultiPolygon | list | pd.Series, 
        buffer_in_meters: float, # in meters
        is_geographic_poly: bool = False
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    polygons = _check_and_collect_polygons_in_list(polygons)
    polygons_out = polygons.copy()

    if is_geographic_poly:
        proj_epsg = []
        for idx, p in enumerate(polygons_out):
            poly_bbox = p.bounds
            if not are_coords_geographic(poly_bbox[2], poly_bbox[3]):
                raise ValueError(f"Polygon at index {idx} is not geographic.")
            proj_epsg.append(get_projected_epsg_code_from_bbox(poly_bbox))
        polygons_out = convert_polygons_crs(p, crs_in=4326, crs_out=proj_epsg)

    for idx, p in enumerate(polygons_out):
        polygons_out[idx] = p.buffer(buffer_in_meters)

    if is_geographic_poly:
        polygons_out = convert_polygons_crs(p, crs_in=proj_epsg, crs_out=4326)
    return polygons_out

# %%