# %% === Import necessary modules
import os
import sys
import logging
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    LOG_CONFIG
)

from psliptools import (
    load_shapefile_polygons,
    get_shapefile_fields
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
        src_path = input(f"{source_type} shapefile name (e.g. {source_type}.shp) or full path: ").strip(' "')
        if not os.path.isabs(src_path):
            src_path = os.path.join(env.folders['inputs'][source_type]['path'], src_path)
        
        shp_fields, shp_types = get_shapefile_fields(src_path)
        print("Shapefile fields and types:")
        for i, (f, t) in enumerate(zip(shp_fields, shp_types)):
            print(f"{i+1}. {f} ({t})")
        sel_shp_field = input(f"Field name containing polygon names (or number from 1 to {len(shp_fields)}): ").strip(' "')
        if sel_shp_field.isdigit() and 1 <= int(sel_shp_field) <= len(shp_fields):
            sel_shp_field = shp_fields[int(sel_shp_field)-1]
    
    prop_df = load_shapefile_polygons(
        shapefile_path=src_path,
        field_name=sel_shp_field,
        poly_bound_geo=study_area_polygon,
        mask_out_poly=True,
        convert_to_geo=True
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
    main()
