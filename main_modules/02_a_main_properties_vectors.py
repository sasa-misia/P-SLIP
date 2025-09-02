# %% === Import necessary modules
import os
import sys
import logging
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    LOG_CONFIG,
    KNOWN_OPTIONAL_STATIC_INPUT_TYPES
)

# Importing necessary modules from psliptools
from psliptools.geometries import (
    load_shapefile_polygons,
    get_shapefile_fields
)

from psliptools.utilities import (
    select_file_prompt,
    select_from_list_prompt
)

# Importing necessary modules from main_modules
from env_init import get_or_create_analysis_environment

# %% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# %% === Methods to import shapefiles with main properties
def obtain_config_idx_and_rel_filename(
        env: AnalysisEnvironment, 
        source_type: str, 
        source_subtype: str=None
    ) -> tuple[AnalysisEnvironment, int, str]:
    idx = 0
    if source_subtype:
        if env.config['inputs'][source_type][0]['settings']: # if the setting dictionary of the first element [0] is not empty, then you should overwrite or add an element to the list
            poss_idx = []
            for i, d in enumerate(env.config['inputs'][source_type]):
                if 'source_subtype' in d['settings'].keys():
                    if d['settings']['source_subtype'] == source_subtype:
                        poss_idx.append(i)
            if len(poss_idx) > 1:
                raise ValueError("Multiple subtypes with the same name were found. Please check the subtype.")
            elif len(poss_idx) ==  1:
                idx = poss_idx[0]
            else:
                idx += len(env.config['inputs'][source_type]) # This must be before the append!
                env.config['inputs'][source_type].append({})
        rel_filename = f"{source_type}_{source_subtype}"
    else:
        rel_filename = f"{source_type}"
    return env, idx, rel_filename

# %% === Main function
def main(source_type: str="land_use", source_subtype: str=None, gui_mode: bool=False, base_dir: str=None):
    if not source_type in KNOWN_OPTIONAL_STATIC_INPUT_TYPES:
        raise ValueError("Invalid source type. Must be one of: " + ", ".join(KNOWN_OPTIONAL_STATIC_INPUT_TYPES))

    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    env, idx_config, rel_filename = obtain_config_idx_and_rel_filename(env, source_type, source_subtype)

    study_area_polygon = env.load_variable(variable_filename='study_area_vars.pkl')['study_area_polygon']

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Shapefile selection ===")
        src_path = select_file_prompt(
            base_dir=env.folders['inputs'][source_type]['path'],
            usr_prompt=f"Name or full path of the {source_type} [subtype: {source_subtype}] shapefile (ex. {source_type}.shp): ",
            src_ext='shp'
        )

        shp_fields, shp_types = get_shapefile_fields(src_path)
        print("\n=== Shapefile fields and types ===")
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

    prop_df['label'] = prop_df['class_name']
    prop_df['standardized_class_id'] = None
    prop_df['parameters_class_id'] = None
    prop_df['info'] = None

    # Write the DataFrame to a CSV file excluding the 'geometry' column
    association_filename = f"{rel_filename}_association.csv"
    csv_path = os.path.join(env.folders['user_control']['path'], association_filename)
    prop_df.drop(columns=['geometry']).to_csv(csv_path, index=False)

    prop_vars = {'prop_df': prop_df}

    _, cust_id = env.add_input_file(file_path=src_path, file_type=source_type, file_subtype=source_subtype)

    env.config['inputs'][source_type][idx_config]['settings'] = {
        'source_mode': 'shapefile',
        'source_field': sel_shp_field,
        'source_subtype': source_subtype,
        'association_filename': association_filename
    }
    env.config['inputs'][source_type][idx_config]['custom_id'] = [cust_id]
    env.collect_input_files(file_type=[source_type], file_subtype=[source_subtype], multi_extension=True)
    
    env.save_variable(variable_to_save=prop_vars, variable_filename=f"{rel_filename}_vars.pkl")
    return prop_vars
    
# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import shapefiles with main properties")
    parser.add_argument("--source_type", type=str, default="land_use", help="Source type (e.g., " + ", ".join(KNOWN_OPTIONAL_STATIC_INPUT_TYPES) + ")")
    parser.add_argument("--source_subtype", type=str, default=None, help="Source subtype (optional)")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for the analysis")
    args = parser.parse_args()

    prop_vars = main(
        source_type=args.source_type, 
        source_subtype=args.source_subtype,
        gui_mode=args.gui_mode, 
        base_dir=args.base_dir
    )

# %%