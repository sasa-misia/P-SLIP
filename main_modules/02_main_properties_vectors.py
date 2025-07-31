# %% === Import necessary modules
import os
import sys
import logging
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    LOG_CONFIG
)

from psliptools import (
    load_shapefile_polygons,
    get_shapefile_fields,
    select_file_prompt,
    select_from_list_prompt
)

from env_init import get_or_create_analysis_environment

#%% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# %% === Methods to import shapefiles with main properties

# %% === Main function
def main(source_type: str="land_uses", gui_mode: bool=False, base_dir: str=None):
    if not source_type in ("soil", "vegetation", "land_uses"):
        raise ValueError("Invalid source type. Must be 'soil', 'vegetation', or 'land_uses'.")

    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    study_area_polygon = env.load_variable(variable_filename='study_area_vars.pkl')['study_area_polygon']

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        src_path = select_file_prompt(
            base_dir=env.folders['inputs'][source_type]['path'],
            usr_prompt=f"Name or full path of the {source_type} shapefile (ex. {source_type}.shp): ",
            src_ext='shp'
        )

        shp_fields, shp_types = get_shapefile_fields(src_path)
        print("\n === Shapefile fields and types ===")
        sel_shp_field = select_from_list_prompt(
            obj_list=shp_fields, 
            obj_type=shp_types, 
            usr_prompt="Select the field:", 
            allow_multiple=False
        )[0]
    
    prop_df = load_shapefile_polygons(
        shapefile_path=src_path,
        field_name=sel_shp_field,
        poly_bound_geo=study_area_polygon,
        mask_out_poly=True,
        convert_to_geo=True,
        points_lim=300000
    )

    prop_vars = {'prop_polygons_df': prop_df}

    _, cust_id = env.add_input_file(file_path=src_path, file_type=source_type)

    env.config['inputs'][source_type][0]['settings'] = {
        'source_mode': 'shapefile',
        'source_field': sel_shp_field
    }
    env.config['inputs'][source_type][0]['custom_id'] = cust_id
    env.collect_input_files(file_type=[source_type], multi_extension=True)
    
    env.save_variable(variable_to_save=prop_vars, variable_filename=f"{source_type}_vars.pkl")
    return prop_vars
    
# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import shapefiles with main properties")
    parser.add_argument("--source_type", type=str, default="land_uses", help="Source type (soil, vegetation, land_uses)")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for the analysis")
    args = parser.parse_args()

    prop_vars = main(
        source_type=args.source_type, 
        gui_mode=args.gui_mode, 
        base_dir=args.base_dir
    )

# %%