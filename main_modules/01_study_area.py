#%% # Import necessary modules
import os
import json
import shapely.geometry as geom
import shapely.ops as ops
import geopandas as gpd
import pandas as pd
import numpy as np

# Importing necessary modules from config
from config import get_analysis_environment, AnalysisEnvironment

# Importing necessary modules from psliptools
from psliptools.utilities import get_raw_path
from psliptools.geometries import (
    load_shapefile_polygons, 
    create_rectangle_polygons, 
    intersect_polygons,
    union_polygons,
    get_polygon_extremes
)

#%% # Define study area functions
def define_study_area_from_shapefile(study_area_fold, study_area_filename, id_field, id_selection, clip_polygons=None):
    """Define study area from a shapefile, optionally clipping with custom polygons."""
    shapefile_path = os.path.join(study_area_fold, study_area_filename)
    study_area_df = load_shapefile_polygons(shapefile_path, field_name=id_field, sel_filter=id_selection)
    if clip_polygons:
        clip_union = ops.unary_union(clip_polygons)
        id_polys = intersect_polygons(study_area_df['geometry'], clip_union)
    study_area_poly = union_polygons(id_polys)
    study_area_extremes = get_polygon_extremes(study_area_poly)
    study_area_vars = {
        'study_area_polygon': study_area_poly,
        'study_area_cls_poly': study_area_df,
        'study_area_cln_poly': study_area_poly,
        'study_area_rem_poly': pd.DataFrame(), # Empty DataFrame for removed areas
        'study_area_extremes': study_area_extremes
    }
    return study_area_vars

def define_study_area_from_rectangles(rectangle_polygons):
    """Define study area from rectangle polygons."""
    rect_polys = rectangle_polygons
    rect_names = [f"Poly {i+1}" for i in range(len(rectangle_polygons))]
    study_area_df = pd.DataFrame({
        'class_name': rect_names,
        'geometry': rect_polys
    })
    study_area_poly = ops.unary_union(rect_polys)
    study_area_extremes = get_polygon_extremes(study_area_poly)
    study_area_vars = {
        'study_area_polygon': study_area_poly,
        'study_area_cls_poly': study_area_df,
        'study_area_cln_poly': study_area_poly,
        'study_area_rem_poly': pd.DataFrame(), # Empty DataFrame for removed areas
        'study_area_extremes': study_area_extremes
    }
    return study_area_vars

def update_env_study_area(env: AnalysisEnvironment, save_path: str, study_area_choices, study_area_vars_file, landuse_vars_file):
    """Update the analysis environment with study area info."""
    env.study_area = {
        "choices": study_area_choices,
        "study_area_vars_file": study_area_vars_file,
        "landuse_vars_file": landuse_vars_file
    }
    env.to_json(save_path)
    return env

#%% # Main function to define the study area
def main(base_dir=None):
    """Main function to define the study area."""
    # --- Initialize environment ---
    env, env_path = get_analysis_environment(base_dir=base_dir)

    # --- User choices section ---
    study_area_file = input("Study area shapefile name (e.g. comuni.shp): ")
    mun_field = input("Municipality field name: ")
    mun_selection = input("Municipalities to select (comma separated): ").split(",")
    use_window = input("Define study area using rectangles? [y/N]: ").strip().lower() == "y"
    rectangle_polygons = []
    rectangle_params = []
    
    if use_window:
        n_rectangles = int(input("How many rectangles? [1]: ") or "1")
        for i in range(n_rectangles):
            print(f"Rectangle {i+1}:")
            lon_min = float(input("  Lon min [°]: "))
            lon_max = float(input("  Lon max [°]: "))
            lat_min = float(input("  Lat min [°]: "))
            lat_max = float(input("  Lat max [°]: "))
            rectangle_params.append((lon_min, lon_max, lat_min, lat_max))
        rectangle_polygons = create_rectangle_polygons(rectangle_params)

    # --- Step 1: Study area definition ---
    if use_window:
        study_area_vars, mun_names = define_study_area_from_rectangles(rectangle_polygons)
    else:
        # If you want to clip with rectangles, pass rectangle_polygons as clip_polygons
        study_area_vars, mun_names = define_study_area_from_shapefile(
            env, study_area_file, mun_field, mun_selection, clip_polygons=rectangle_polygons if rectangle_polygons else None
        )

    # Initialize removed_areas if not present
    if 'removed_areas' not in env:
        env['removed_areas'] = []

    # --- Save all choices in env.study_area ---
    study_area_choices = {
        "study_area_file": study_area_file,
        "mun_field": mun_field,
        "mun_selection": mun_selection,
        "use_window": use_window,
        "rectangle_params": rectangle_params,
        "mun_names": mun_names
    }
    study_area_vars_file = os.path.join(fold_var, 'study_area_vars.pkl')
    # Removed landuse_vars_file as it is no longer used
    update_env_study_area(env, study_area_choices, study_area_vars_file, None)

    # Save env
    with open(env_path, 'w', encoding='utf-8') as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

    print("Study area updated. Land use import and removal are now handled in separate modules.")

if __name__ == "__main__":
    main()