# %% === Import necessary modules
import os
import sys
import logging
import argparse
import pandas as pd
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    LOG_CONFIG
)

# Importing necessary modules from psliptools
from psliptools.geometries import (
    load_shapefile_polygons,
    get_rectangle_parameters,
    create_rectangle_polygons, 
    intersect_polygons,
    union_polygons,
    get_polygon_extremes,
    get_shapefile_fields,
    get_shapefile_field_values
)

from env_init import get_or_create_analysis_environment

# %% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# %% === Study Area methods
def define_study_area_from_shapefile(shapefile_path, id_field, id_selection, clip_polygons=None):
    """Define study area from a shapefile, optionally clipping with custom polygons."""
    study_area_df = load_shapefile_polygons(
        shapefile_path=shapefile_path, 
        field_name=id_field, 
        sel_filter=id_selection,
        convert_to_geo=True
    )
    id_polys = study_area_df['geometry']
    if clip_polygons:
        id_polys = intersect_polygons(id_polys, clip_polygons, clean_empty=True)
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
    study_area_poly = union_polygons(rect_polys)
    study_area_extremes = get_polygon_extremes(study_area_poly)
    study_area_vars = {
        'study_area_polygon': study_area_poly,
        'study_area_cls_poly': study_area_df,
        'study_area_cln_poly': study_area_poly,
        'study_area_rem_poly': pd.DataFrame(), # Empty DataFrame for removed areas
        'study_area_extremes': study_area_extremes
    }
    return study_area_vars

# %% === Main function to define the study area
def main(gui_mode: bool=False, base_dir: str=None) -> Dict[str, object]:
    """Main function to define the study area."""
    src_type = 'study_area'

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=True)

    # --- User choices section ---
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        use_window = input("Define study area using rectangles? [y/N]: ").strip().lower() == "y"
        if not(use_window):
            src_mode = 'shapefile'
            src_path = input("Study area shapefile name (e.g. study_area.shp) or full path: ").strip(' "')
            if not os.path.isabs(src_path):
                src_path = os.path.join(env.folders['inputs']['study_area']['path'], src_path)

            shp_fields, shp_types = get_shapefile_fields(src_path)
            print("Shapefile fields and types:")
            for i, (f, t) in enumerate(zip(shp_fields, shp_types)):
                print(f"{i+1}. {f} ({t})")
            cls_fld = input(f"Field name containing polygon names (or number from 1 to {len(shp_fields)}): ").strip(' "')
            if cls_fld.isdigit() and 1 <= int(cls_fld) <= len(shp_fields):
                cls_fld = shp_fields[int(cls_fld)-1]

            shp_field_vals = get_shapefile_field_values(src_path, cls_fld, sort=True)
            print("Available classes:")
            for i, val in enumerate(shp_field_vals):
                print(f"{i+1}. {val}")
            cls_sel = [x.strip(' "') for x in input("Classes to select (also multiple, comma or semicolon separated): ").replace(',', ';').split(';')]
            cls_sel = sorted(set(shp_field_vals[int(x)-1] if x.isdigit() and 1 <= int(x) <= len(shp_field_vals) else x for x in cls_sel))
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
        _, cust_id = env.add_input_file(file_path=src_path, file_type='study_area')
        study_area_vars = define_study_area_from_shapefile(
            shapefile_path=src_path, 
            id_field=cls_fld, 
            id_selection=cls_sel
        )

    env.config['inputs'][src_type][0]['settings'] = {
        'source_mode': src_mode,
        'source_field': cls_fld,
        'source_selection': cls_sel
    }
    env.config['inputs'][src_type][0]['custom_id'] = [cust_id]
    env.collect_input_files(file_type=[src_type], multi_extension=True)
    
    env.save_variable(variable_to_save=study_area_vars, variable_filename="study_area_vars.pkl")
    return study_area_vars

 # %% === Command line interface
if __name__ == "__main__":
    # Command line interface
    parser = argparse.ArgumentParser(description="Define the study area for the analysis.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    args = parser.parse_args()

    study_area_vars = main(base_dir=args.base_dir, gui_mode=args.gui_mode)

# %%
