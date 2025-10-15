# %% === Import necessary modules
import os
import sys
import argparse
import pandas as pd
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
# from config import (
# )

# Importing necessary modules from psliptools
from psliptools.geometries import (
    load_shapefile_polygons,
    get_rectangle_parameters,
    create_rectangle_polygons,
    union_polygons,
    get_polygon_extremes,
    get_shapefile_fields,
    get_shapefile_field_values
)

from psliptools.utilities import (
    select_file_prompt,
    select_from_list_prompt
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger(__name__)
logger.info("=== Import or Create Study Area ===")

# %% === Study Area methods
REM_POLY_DF = pd.DataFrame(columns=['type', 'subtype', 'class_name', 'geometry']) # Empty DataFrame for removed areas

def define_study_area_from_shapefile(shapefile_path, id_field, id_selection):
    """Define study area from a shapefile, optionally clipping with custom polygons."""
    study_area_df = load_shapefile_polygons(
        shapefile_path=shapefile_path, 
        field_name=id_field, 
        sel_filter=id_selection,
        convert_to_geo=True
    )
    id_polys = study_area_df['geometry']
    study_area_poly = union_polygons(id_polys)
    study_area_extremes = get_polygon_extremes(study_area_poly)
    study_area_vars = {
        'study_area_polygon': study_area_poly,
        'study_area_cls_poly': study_area_df,
        'study_area_cln_poly': study_area_poly,
        'study_area_rem_poly': REM_POLY_DF,
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
        'study_area_rem_poly': REM_POLY_DF,
        'study_area_extremes': study_area_extremes
    }

    return study_area_vars

# %% === Main function to define the study area
def main(
        base_dir: str=None,
        gui_mode: bool=False
    ) -> Dict[str, object]:
    """Main function to define the study area."""
    src_type = 'study_area'

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=True)

    # --- User choices section ---
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        use_window = input("Define study area using rectangles? [y/N]: ").strip().lower() == "y"
        if use_window:
            src_mode = 'rectangle'
            src_path = None
            cls_fld = None
            cls_sel = None
            n_rectangles = int(input("How many rectangles? [1]: ") or "1")
            rec_polys = create_rectangle_polygons(get_rectangle_parameters(n_rectangles))
        else:
            src_mode = 'shapefile'
            print("\n=== Shapefile selection ===")
            src_path = select_file_prompt(
                base_dir=env.folders['inputs']['study_area']['path'],
                usr_prompt=f"Name or full path of the {src_type} shapefile (ex. {src_type}.shp): ",
                src_ext='shp'
            )

            shp_fields, shp_types = get_shapefile_fields(src_path)
            print("\n=== Shapefile fields and types ===")
            cls_fld = select_from_list_prompt(
                obj_list=shp_fields, 
                obj_type=shp_types, 
                usr_prompt="Select the field:", 
                allow_multiple=False
            )[0]

            shp_field_vals = get_shapefile_field_values(src_path, cls_fld, sort=True)
            print("\n=== Shapefile field classes ===")
            cls_sel = select_from_list_prompt(
                obj_list=shp_field_vals,
                usr_prompt=f"Select the class(es) inside the field ({cls_fld}):", 
                allow_multiple=True
            )

    # --- Step 1: Study area definition ---
    if use_window:
        study_area_vars = define_study_area_from_rectangles(rec_polys)
    else:
        # If you want to clip with rectangles, pass something as clip_polygons
        _, cust_id = env.add_input_file(file_path=src_path, file_type=src_type)
        study_area_vars = define_study_area_from_shapefile(
            shapefile_path=src_path, 
            id_field=cls_fld, 
            id_selection=cls_sel
        )

    env.config['inputs'][src_type][0]['settings'] = {
        'source_mode': src_mode,
        'source_field': cls_fld,
        'source_selection': cls_sel,
        'source_refined': False
    }
    env.config['inputs'][src_type][0]['custom_id'] = [cust_id]
    env.collect_input_files(file_type=[src_type], multi_extension=True)

    env.save_variable(variable_to_save=study_area_vars, variable_filename=f"{src_type}_vars.pkl")
    
    return study_area_vars

 # %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the study area for the analysis.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    args = parser.parse_args()

    study_area_vars = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode
    )

# %%
