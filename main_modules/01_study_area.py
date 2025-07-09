#%% # Import necessary modules
import os
import shapely.ops as ops
import pandas as pd
import argparse
from typing import Dict

# Importing necessary modules from config
from config import get_analysis_environment, AnalysisEnvironment, save_variable

# Importing necessary modules from psliptools
from psliptools.utilities import get_raw_path
from psliptools.geometries import (
    load_shapefile_polygons,
    get_rectangle_parameters,
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

#%% # Main function to define the study area
def main(gui_mode=False, base_dir=None) -> Dict[str, object]:
    """Main function to define the study area."""
    # --- Initialize environment ---
    env, _ = get_analysis_environment(base_dir=base_dir)

    src_type = 'study_area'

    # --- User choices section ---
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        use_window = input("Define study area using rectangles? [y/N]: ").strip().lower() == "y"
        if not(use_window):
            src_mode = 'shapefile'
            src_path = input("Study area shapefile name (e.g. study_area.shp) or full path: ")
            cls_fld = input("Field name containing class names: ")
            cls_sel = input("Classes to select (comma separated): ").split(",")
        else:
            src_mode = 'rectangle'
            src_path = None
            cls_fld = None
            cls_sel = None
            n_rectangles = int(input("How many rectangles? [1]: ") or "1")
            rec_polys = create_rectangle_polygons(get_rectangle_parameters(n_rectangles))

    # --- Step 1: Study area definition ---
    if use_window:
        study_area_vars = define_study_area_from_rectangles(rec_polys)
    else:
        # If you want to clip with rectangles, pass rectangle_polygons as clip_polygons
        study_area_vars = define_study_area_from_shapefile(
            study_area_filename=src_path, 
            id_field=cls_fld, id_selection=cls_sel
        )

    env.user_control[src_type]['source_mode'] = src_mode
    env.user_control[src_type]['source_type'] = src_type
    env.user_control[src_type]['source_field'] = cls_fld
    env.user_control[src_type]['source_selection'] = cls_sel
    
    save_variable(analysis_env=env, variable_to_save=study_area_vars, filename="study_area_vars.json", var_type="study_area")
    return study_area_vars

if __name__ == "__main__":
    # Command line interface
    parser = argparse.ArgumentParser(description="Define the study area for the analysis.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    args = parser.parse_args()

    study_area_vars = main(base_dir=args.base_dir, gui_mode=False)