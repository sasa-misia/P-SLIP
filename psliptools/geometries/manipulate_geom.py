# %% === Import necessary modules
import pandas as pd
import numpy as np
import shapely.geometry as geom
import shapely.ops as ops
import warnings
from scipy.spatial import KDTree

from ..rasters.coordinates import convert_coords_from_list, are_coords_geographic, get_projected_epsg_code_from_bbox

# %% === Helper function to process both Polygon and MultiPolygon geometries with parameters
def _process_polygon_geometries(
        polygon: geom.Polygon | geom.MultiPolygon,
        process_function: callable,
        *args,
        **kwargs
    ) -> list:
    """
    Process both Polygon and MultiPolygon geometries using a unified approach with parameters.
    
    Args:
        polygon (geom.Polygon | geom.MultiPolygon): Polygon or MultiPolygon to process
        process_function (callable): Function to apply to each individual polygon geometry
        *args: Positional arguments to pass to process_function
        **kwargs: Keyword arguments to pass to process_function
    
    Returns:
        list(shapely.geometry): List of processed geometries
    """
    if isinstance(polygon, geom.Polygon):
        return [process_function(polygon, *args, **kwargs)]
    elif isinstance(polygon, geom.MultiPolygon):
        return [process_function(geom_poly, *args, **kwargs) for geom_poly in polygon.geoms]
    else:
        raise ValueError("Input must be a Polygon or MultiPolygon")

# %% === Function to check the geometry of a polygon
def _check_polygon_is_valid(
    polygon: geom.Polygon | geom.MultiPolygon
    ) -> bool:
    """
    Check the geometry of a polygon.

    Args:
        polygon (geom.Polygon | geom.MultiPolygon): Polygon or MultiPolygon to check.

    Returns:
        bool: True if the polygon is valid, False otherwise.

    Raises:
        ValueError: If the input is not a Polygon or MultiPolygon.
    """
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
    """
    Check the geometry of a polygon and fix it if necessary.

    Args:
        polygon (geom.Polygon | geom.MultiPolygon): Polygon or MultiPolygon to check.

    Returns:
        (geom.Polygon | geom.MultiPolygon): Fixed polygon or multipolygon.

    Raises:
        ValueError: If the input is not a Polygon or MultiPolygon.
    """
    if not isinstance(polygon, (geom.Polygon, geom.MultiPolygon)):
        raise ValueError("Input must be a shapely Polygon or MultiPolygon.")
    if not _check_polygon_is_valid(polygon):
        polygon = polygon.buffer(0)
        if not _check_polygon_is_valid(polygon):
            raise ValueError("Tried to fix polygon geometry, but it is still invalid.")
        
    return polygon

# %% === Function to check content of a list of supposed polygons (or MultiPolygon)
def _check_and_collect_polygons_in_list(
        polygons: geom.Polygon | geom.MultiPolygon | list[geom.Polygon | geom.MultiPolygon] | pd.Series,
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Check that a list of polygons (or MultiPolygon) is valid and if not, convert it.

    Args:
        polygons (list | pd.Series): List of polygons (or MultiPolygon) to check.
    
    Returns:
        list(geom.Polygon | geom.MultiPolygon): List of polygons (or MultiPolygon).

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
        polygons[idx] = _check_and_fix_polygon(p)

    return polygons

# %% === Helper function to remove consecutive duplicates
def _remove_consecutive_duplicates(
        x_coords: list[float | int],
        y_coords: list[float | int],
        keep_last_point: bool = True
    ) -> tuple[list[float | int], list[float | int]]:
    """
    Remove consecutive duplicates from a list of x and y coordinates.

    Args:
        x_coords (list): List of x coordinates.
        y_coords (list): List of y coordinates.
        keep_last_point (bool, optional): Whether to keep the last point (in case it is equal to the first). Defaults to True.

    Returns:
        tuple(list, list): Tuple containing the x and y coordinates without consecutive duplicates (if any). 
            Note: if the last point is equal to the first point, it is removed.
    """
    if len(x_coords) <= 1:
        return x_coords, y_coords
    
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")
    
    unique_x, unique_y = [x_coords[0]], [y_coords[0]]
    for i in range(1, len(x_coords)):
        if x_coords[i] != x_coords[i-1] or y_coords[i] != y_coords[i-1]: # Just consecutive points are duplicates! Be careful...
            unique_x.append(x_coords[i])
            unique_y.append(y_coords[i])
    
    # Check if first and last points are the same (closed polygon)
    if not keep_last_point:
        if len(unique_x) > 1 and unique_x[0] == unique_x[-1] and unique_y[0] == unique_y[-1]:
            unique_x = unique_x[:-1] # Remove last point (equal to first point)
            unique_y = unique_y[:-1] # Remove last point (equal to first point)

    return unique_x, unique_y

# %% === Function to create a single polygon from a list of polygons
def union_polygons(
        polygons: list[geom.Polygon | geom.MultiPolygon] | pd.Series,
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
        polygons: geom.Polygon | geom.MultiPolygon | list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        mask: geom.Polygon | geom.MultiPolygon | list[geom.Polygon | geom.MultiPolygon] | pd.Series,
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
        warnings.warn("The intersection of the polygons with the mask is empty.", stacklevel=2)

    return intersected_poly

# %% === Function to subtract a polygon from a list of polygons
def subtract_polygons(
        polygons: geom.Polygon | geom.MultiPolygon | list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        subtract_mask: geom.Polygon | geom.MultiPolygon | list[geom.Polygon | geom.MultiPolygon] | pd.Series,
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

    subtracted_polygon = [p.difference(subtract_union) for p in polygons]

    return subtracted_polygon

# %% === Function to obtain list of coordinates (exterior and interiors) from a polygon
def get_poly_external_and_internal(
        polygon: geom.Polygon,
        remove_duplicates: bool = False
    ) -> tuple[list, list]:
    """
    Obtain list of coordinates from a polygon

    Args:
        polygon (geom.Polygon): Polygon to obtain coordinates from.
        remove_duplicates (bool, optional): Whether to remove duplicate coordinates. Defaults to True.

    Returns:
        tuple[list, list]: List of x and y coordinates (first element inside each list is the exterior and the rest are the interiors).
    """
    if not isinstance(polygon, geom.Polygon):
        raise ValueError("Input must be a shapely Polygon, not MultiPolygon or something else.")
    
    x_coords_poly_list, y_coords_poly_list = [], []
    
    # Process exterior
    x_exterior = list(polygon.exterior.coords.xy[0])
    y_exterior = list(polygon.exterior.coords.xy[1])
    
    if remove_duplicates:
        x_exterior, y_exterior = _remove_consecutive_duplicates(x_exterior, y_exterior)
    
    x_coords_poly_list.append(x_exterior)
    y_coords_poly_list.append(y_exterior)
    
    # Process interiors
    for interior in polygon.interiors:
        x_interior = list(interior.coords.xy[0])
        y_interior = list(interior.coords.xy[1])
        
        if remove_duplicates:
            x_interior, y_interior = _remove_consecutive_duplicates(x_interior, y_interior)
        
        x_coords_poly_list.append(x_interior)
        y_coords_poly_list.append(y_interior)
    
    return x_coords_poly_list, y_coords_poly_list

# %% === Function to create a polygon from a list of coordinates
def create_polygon_from_coord_lists(
        x_coords: list[list[float | int] | np.ndarray],
        y_coords: list[list[float | int] | np.ndarray],
        remove_duplicates: bool = True
    ) -> geom.Polygon:
    """
    Create a polygon from a list of x and y coordinates.

    Args:
        x_coords (list): List of x coordinates, where each element of the list contains an array (or list) of x coordinates (first element inside the list is the exterior and the rest are the interiors).
        y_coords (list): List of y coordinates, where each element of the list contains an array (or list) of y coordinates (first element inside the list is the exterior and the rest are the interiors).
        remove_duplicates (bool, optional): Whether to remove duplicate coordinates. Defaults to True.

    Returns:
        geom.Polygon: Polygon created from x and y coordinates.
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")
    
    # Process exterior
    x_exterior = x_coords[0]
    y_exterior = y_coords[0]
    
    if remove_duplicates:
        x_exterior, y_exterior = _remove_consecutive_duplicates(x_exterior, y_exterior, keep_last_point=False)
    
    if len(x_exterior) < 3: # Always after removing duplicates
        warnings.warn("Polygon has less than 3 points, it will be an empty Polygon.", stacklevel=2)
        return geom.Polygon()
    
    exterior = [(x, y) for x, y in zip(x_exterior, y_exterior)]
    
    # Process interiors
    interiors = []
    if len(x_coords) > 1:
        for idx, (x_points, y_points) in enumerate(zip(x_coords[1:], y_coords[1:])):
            if len(x_points) != len(y_points):
                raise ValueError(f"x_coords and y_coords at index {idx} must have the same length.")
            
            if remove_duplicates:
                x_points, y_points = _remove_consecutive_duplicates(x_points, y_points, keep_last_point=False)

            if len(x_points) < 3: # Always after removing duplicates
                warnings.warn(f"Polygon with index {idx} has less than 3 points, ignoring.", stacklevel=2)
                continue
            
            interiors.append([(x, y) for x, y in zip(x_points, y_points)])
    
    if interiors:
        out_polygon = geom.Polygon(exterior, holes=interiors)
    else:
        out_polygon = geom.Polygon(exterior)

    return out_polygon

# %% === Function to convert a single polygon from one CRS to another
def convert_single_polygon_crs(
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
    x_coords_poly_in, y_coords_poly_in = get_poly_external_and_internal(polygon)
    x_coords_poly_out, y_coords_poly_out = convert_coords_from_list(crs_in, crs_out, x_coords_poly_in, y_coords_poly_in)
    polygon_out = create_polygon_from_coord_lists(x_coords_poly_out, y_coords_poly_out)

    return polygon_out

# %% === Function to convert multiple polygons from one CRS to another
def convert_polygons_crs(
        polygons: geom.Polygon | geom.MultiPolygon | list[geom.Polygon | geom.MultiPolygon] | pd.Series,
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
        if isinstance(p, (geom.Polygon, geom.MultiPolygon)):
            def _convert_single_geom(geom_poly):
                return _check_and_fix_polygon(convert_single_polygon_crs(geom_poly, crs_in_list[idx], crs_out_list[idx]))
            
            converted_geoms = _process_polygon_geometries(p, _convert_single_geom)
            
            if len(converted_geoms) == 1:
                polygons_out[idx] = converted_geoms[0]
            else:
                polygons_out[idx] = geom.MultiPolygon(converted_geoms)
        else:
            raise ValueError("each element of polygons must be a shapely Polygon or MultiPolygon.")
        
    return polygons_out

# %% === Function to add a buffer (in meters) to a polygon
def add_buffer_to_polygons(
        polygons: geom.Polygon | geom.MultiPolygon | list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        buffer_in_meters: float, # in meters
        is_geographic_poly: bool = False
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Add a buffer (in meters) to a polygon (or MultiPolygon, or list of polygons).

    Args:
        polygons (geom.Polygon | geom.MultiPolygon | list | pd.Series): Polygons (or MultiPolygon) to buffer.
        buffer_in_meters (float): The buffer to add (in meters).
        is_geographic_poly (bool, optional): True if the input polygon is geographic. Defaults to False.

    Returns:
        list: List of buffered polygons (or MultiPolygons).
    """
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

# %% === Function to check if polygons are properly aligned
def check_and_report_polygons_alignment(
        polygons: list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        tolerance: float = 1e-7
    ) -> dict:
    """
    Check if polygons are properly aligned without gaps or overlaps.
    
    Args:
        polygons (list | pd.Series): List of polygons to check
        tolerance (float, optional): Tolerance for considering boundaries aligned (in coordinate units) (default: 1e-7)
        
    Returns:
        dict: Alignment check results
    """
    polygons = _check_and_collect_polygons_in_list(polygons)
    
    alignment_results = {
        'polygons_analyzed': len(polygons),
        'area_tolerance': tolerance,
        'aligned': True,
        'overlap_area': None,
        'misaligned_edges': []
    }
    
    # Check for gaps and overlaps
    union_all = ops.unary_union(polygons)
    union_area = union_all.area
    total_area = sum(p.area for p in polygons)
    
    if abs(total_area - union_area) > tolerance:
        alignment_results['aligned'] = False
        alignment_results['overlap_area'] = total_area - union_area
    
        # Check individual polygon boundaries
        seen_pairs = set()
        for i, poly1 in enumerate(polygons):
            for j, poly2 in enumerate(polygons):
                if i != j and (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                    seen_pairs.add((i, j))
                    
                    if poly1.touches(poly2):
                        # Check if boundaries are properly aligned
                        boundary1 = poly1.boundary
                        boundary2 = poly2.boundary
                        
                        if not boundary1.intersection(boundary2).is_empty:
                            # Boundaries intersect, check if they're properly aligned
                            intersection = boundary1.intersection(boundary2)
                            alignment_results['misaligned_edges'].append((i, j, intersection.length))

    return alignment_results

# %% === Function to know if polygons have intersections and if so, which ones
def detect_intersections_between_polygons(
        polygons: list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        tolerance: float = 1e-7,
        start_indices_from_1: bool = False
    ) -> list[list[int]]:
    """
    Check if polygons have intersections and if so, which ones.

    Args:
        polygons (list | pd.Series): Polygons to check.
        start_indices_from_1 (bool, optional): True if indices of polygons should start from 1. Defaults to False (start from 0).

    Returns:
        list[list[int]]: List of lists of indices of polygons that intersect and overlap.
    """
    polygons = _check_and_collect_polygons_in_list(polygons)

    polygons_alignment = check_and_report_polygons_alignment(polygons, tolerance=tolerance)

    if polygons_alignment['aligned']:
        poly_intersections_indices = [[] for _ in range(len(polygons))]
    else:
        poly_intersections_indices = []
        for idx, poly in enumerate(polygons):
            overlap_mask = poly.overlaps(polygons)
            within_mask = poly.within(polygons)
            contain_mask = poly.contains(polygons)

            intersect_mask = overlap_mask | within_mask | contain_mask
            intersect_mask[idx] = False

            if overlap_mask.any() or within_mask.any():
                if start_indices_from_1:
                    poly_intersections_indices.append((np.where(intersect_mask)[0] + 1).tolist())
                else:
                    poly_intersections_indices.append(np.where(intersect_mask)[0].tolist())
            else:
                poly_intersections_indices.append([])

    return poly_intersections_indices

# %% === Function to round coordinates of a single polygon
def round_single_polygon(
        polygon: geom.Polygon,
        tolerance: float = 1e-7,
        remove_duplicates: bool = True
    ) -> geom.Polygon:
    """
    Round coordinates of a single polygon to the specified tolerance.
    
    Args:
        polygon (geom.Polygon): Polygon to round
        tolerance (float, optional): Tolerance for rounding (default: 1e-7)
        remove_duplicates (bool, optional): Whether to remove duplicate coordinates (default: True)
    
    Returns:
        geom.Polygon: Rounded polygon
    """
    # Use existing function to get coordinates and create rounded polygon
    x_coords_list, y_coords_list = get_poly_external_and_internal(polygon)
    x_coords_round_list, y_coords_round_list = [], []
    for x_coords, y_coords in zip(x_coords_list, y_coords_list):
        x_coords_round = [round(x / tolerance) * tolerance for x in x_coords]
        y_coords_round = [round(y / tolerance) * tolerance for y in y_coords]
        x_coords_round_list.append(x_coords_round)
        y_coords_round_list.append(y_coords_round)

    poly_out = create_polygon_from_coord_lists(
        x_coords_round_list,
        y_coords_round_list, 
        remove_duplicates=remove_duplicates
    )
    
    return poly_out

# %% === Function to round coordinates of polygons
def round_polygons(
        polygons: list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        tolerance: float = 1e-7,
        remove_duplicates: bool = True
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Round coordinates of polygons to the specified tolerance.
    
    Args:
        polygons (list | pd.Series): List of polygons to round
        tolerance (float, optional): Tolerance for rounding coordinates (default: 1e-7)
        
    Returns:
        list(geom.Polygon | geom.MultiPolygon): List of rounded polygons
    """
    polygons = _check_and_collect_polygons_in_list(polygons)
    rounded_polygons = []
    
    for poly in polygons:
        if isinstance(poly, (geom.Polygon, geom.MultiPolygon)):
            rounded_geoms = _process_polygon_geometries(
                poly, 
                round_single_polygon,
                tolerance=tolerance, 
                remove_duplicates=remove_duplicates
            )
            
            if len(rounded_geoms) == 1:
                rounded_poly = rounded_geoms[0]
            else:
                rounded_poly = geom.MultiPolygon(rounded_geoms)
            
            # Ensure the polygon is valid after rounding
            rounded_poly = _check_and_fix_polygon(rounded_poly)
            rounded_polygons.append(rounded_poly)
        else:
            rounded_polygons.append(poly)
    
    return rounded_polygons

# %% === Function to collect vertices from a single polygon
def collect_single_polygon_vertices(
        polygon: geom.Polygon,
        all_vertices: list,
        remove_duplicates: bool = True
    ) -> None:
    """
    Collect vertices from a single polygon into a given list.
    
    Args:
        polygon (geom.Polygon): Polygon to collect vertices from
        all_vertices (list): List to append vertices to (automatically updated, no return needed)
        remove_duplicates (bool, optional): Whether to remove duplicate coordinates (default: True)
    
    Returns:
        None
    """
    # Use existing function to get coordinates
    x_coords_list, y_coords_list = get_poly_external_and_internal(polygon, remove_duplicates=remove_duplicates)
    all_vertices.extend(zip(x_coords_list[0], y_coords_list[0]))
    for x_coords, y_coords in zip(x_coords_list[1:], y_coords_list[1:]):
        all_vertices.extend(zip(x_coords, y_coords)) # all_vertices is a list, thus mutable -> no need to return

# %% === Function to snap a single polygon
def snap_single_polygon(
        polygon: geom.Polygon,
        all_vertices_array: np.ndarray,
        kdtree: KDTree,
        remove_duplicates: bool = True
    ) -> geom.Polygon:
    """
    Snap a single polygon's vertices to the nearest collected vertex.
    
    Args:
        polygon (geom.Polygon): Polygon to snap
        all_vertices_array (np.ndarray): Array of all vertices to snap to
        kdtree (KDTree): KDTree for fast nearest neighbor search
        remove_duplicates (bool, optional): Whether to remove duplicate coordinates (default: True)
    
    Returns:
        geom.Polygon: Snapped polygon
    """
    # Use existing function to get coordinates
    x_coords_list, y_coords_list = get_poly_external_and_internal(polygon, remove_duplicates=remove_duplicates)
    
    # Snap exterior
    exterior_coords = list(zip(x_coords_list[0], y_coords_list[0]))
    if all_vertices_array.size > 0:
        _, indices = kdtree.query(exterior_coords)
        snapped_exterior = [all_vertices_array[idx] for idx in indices]
    else:
        snapped_exterior = exterior_coords
    
    # Snap interiors
    snapped_interiors = []
    for x_coords, y_coords in zip(x_coords_list[1:], y_coords_list[1:]):
        interior_coords = list(zip(x_coords, y_coords))
        if all_vertices_array.size > 0:
            _, indices = kdtree.query(interior_coords)
            snapped_interior = [all_vertices_array[idx] for idx in indices]
        else:
            snapped_interior = interior_coords
        snapped_interiors.append(snapped_interior)
    
    # Create polygon from snapped coordinates
    snapped_exterior_x = [coord[0] for coord in snapped_exterior]
    snapped_exterior_y = [coord[1] for coord in snapped_exterior]
    
    snapped_interiors_x = []
    snapped_interiors_y = []
    for interior in snapped_interiors:
        snapped_interiors_x.append([coord[0] for coord in interior])
        snapped_interiors_y.append([coord[1] for coord in interior])

    poly_out = create_polygon_from_coord_lists(
        [snapped_exterior_x] + snapped_interiors_x,
        [snapped_exterior_y] + snapped_interiors_y,
        remove_duplicates=remove_duplicates
    )
    
    return poly_out

# %% === Function to align polygons
def align_polygons(
        polygons: list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        tolerance: float = 1e-7,
        method: str = 'snap'
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Align polygons by snapping vertices to common coordinates or resolving small gaps/overlaps.
    
    Args:
        polygons (list | pd.Series): List of polygons to align
        tolerance (float, optional): It depends on the method and it is in coordinate units (default: 1e-7). 
            - 'snap': Maximum distance for snapping vertices
            - 'buffer': Buffer size to remove from polygons
            - 'round': Tolerance for rounding coordinates
        method (str, optional): Alignment method - 'snap' (snap vertices) (default), 'buffer' (use small buffer), or 'round' (round coordinates)
    
    Returns:
        list(geom.Polygon | geom.MultiPolygon): Aligned polygons
    """
    polygons = _check_and_collect_polygons_in_list(polygons)
    aligned_polygons = []
    
    if method == 'snap':
        # Collect all vertices from all polygons
        all_vertices = []
        for poly in polygons:
            poly_rounded = round_polygons([poly], tolerance)[0]
            
            # Use the new helper function with parameters
            _process_polygon_geometries(
                poly_rounded, 
                collect_single_polygon_vertices, 
                all_vertices, 
                remove_duplicates=True
            )
        
        # Create KDTree for fast nearest neighbor search
        if all_vertices:
            all_vertices_array = np.array(all_vertices)
            kdtree = KDTree(all_vertices_array)
        else:
            raise ValueError("All polygons have no vertices")
        
        # Snap each polygon's vertices to the nearest collected vertex
        for poly in polygons:
            snapped_geoms = _process_polygon_geometries(
                poly,
                snap_single_polygon,
                all_vertices_array,
                kdtree,
                remove_duplicates=True
            )
            
            if len(snapped_geoms) == 1:
                aligned_poly = snapped_geoms[0]
            else:
                aligned_poly = geom.MultiPolygon(snapped_geoms)
            
            aligned_polygons.append(aligned_poly)
    
    elif method == 'buffer':
        # Use a very small buffer to clean up boundaries (this will slightly reduce area of polygons, according to tolerance)
        for poly in polygons:
            aligned_poly = poly.buffer(-tolerance)
            aligned_polygons.append(aligned_poly)

    elif method == 'round':
        # Round coordinates
        aligned_polygons = round_polygons(polygons, tolerance)
    
    else:
        raise ValueError("method must be 'snap' or 'buffer'")
    
    return aligned_polygons

# %% === Function to resolve polygon intersections based on priority
def resolve_polygons_intersections(
        polygons: list[geom.Polygon | geom.MultiPolygon] | pd.Series,
        priority: list[int] = None,
        tolerance: float = 1e-7
    ) -> list[geom.Polygon | geom.MultiPolygon]:
    """
    Resolve intersections between polygons by removing overlapping areas from lower priority polygons.
    
    Args:
        polygons (list | pd.Series): List of polygons to process
        priority (list[int], optional): List of numbers indicating priority order (lower number = higher priority).
            If None, the order in the list is used (first element has highest priority).
        tolerance (float, optional): Tolerance for intersections (default: 1e-7)
    
    Returns:
        list(geom.Polygon | geom.MultiPolygon): Processed polygons with intersections resolved, according to priority and accepted tolerance
    """
    polygons = _check_and_collect_polygons_in_list(polygons)
    processed_polygons = [None] * len(polygons)
    
    if priority is None:
        priority_order = list(range(len(polygons)))
    else:
        if not isinstance(priority, list):
            raise ValueError("priority_indices must be a list.")
        if not all(isinstance(x, int) for x in priority):
            raise ValueError("All elements of priority_indices must be integers.")
        if len(priority) != len(polygons):
            raise ValueError("priority list must have the same length as polygons list.")
        priority_order = priority
    
    # Sort polygons by priority (highest to lowest - h2l)
    sorted_indices_h2l = sorted(range(len(priority_order)), key=lambda i: priority_order[i], reverse=False)
    sorted_polygons_h2l = [polygons[i] for i in sorted_indices_h2l]
    
    # Process from highest priority to lowest
    for i in range(len(sorted_polygons_h2l)):
        current_poly = sorted_polygons_h2l[i]
        higher_priority_polys = sorted_polygons_h2l[:i] # Last element (i) not included
        
        if higher_priority_polys:
            intersect_mask = current_poly.overlaps(higher_priority_polys) | current_poly.within(higher_priority_polys) | current_poly.contains(higher_priority_polys)
            
            # Remove intersection with lower priority polygons from current polygon
            if intersect_mask.any():
                current_poly = subtract_polygons(current_poly, [x for x, y in zip(higher_priority_polys, intersect_mask) if y])[0]
        
        sorted_polygons_h2l[i] = current_poly
    
    # Restore original order
    for current_idx, original_idx in enumerate(sorted_indices_h2l):
        processed_polygons[original_idx] = sorted_polygons_h2l[current_idx]
    
    polygons_alignment = check_and_report_polygons_alignment(processed_polygons, tolerance=tolerance)
    if not polygons_alignment['aligned']:
        warnings.warn("After removing intersections, polygons are not aligned. Trying to remove a buffer to align polygons.", stacklevel=2)
        processed_polygons = align_polygons(processed_polygons, tolerance=tolerance, method='buffer')
        polygons_alignment = check_and_report_polygons_alignment(processed_polygons, tolerance=tolerance)
        
        if not polygons_alignment['aligned']:
            raise ValueError(f"Polygons could not be aligned if you use this tolerance ({tolerance}). Please check the input polygons or modify the tolerance.")
    
    return processed_polygons

# %%