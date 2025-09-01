# %% === Import necessary modules
import os
import sys
import logging
import pandas as pd
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
    subtract_polygons
)

from psliptools.utilities import (
    select_from_list_prompt
)

# Importing necessary modules from main_modules
from env_init import get_or_create_analysis_environment

# %% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'],
                    datefmt=LOG_CONFIG['date_format'])

# %% === Methods to subtract polygons from study area
def subtract_polygons_from_study_area(
    study_area_dict: dict,
    rem_poly_df: pd.DataFrame,
    labels: list,
    type: str,
    subtype: str=None
    ) -> dict:
    poly_sel_indices = rem_poly_df['class_name'].isin(labels)

    polygons_to_remove = rem_poly_df.loc[poly_sel_indices, 'geometry']

    study_area_subtracted_poly_list = subtract_polygons(study_area_dict['study_area_cln_poly'], polygons_to_remove)
    if len(study_area_subtracted_poly_list) != 1:
        logging.error("Polygon subtraction resulted in multiple or no polygons.")
        raise ValueError("Polygon subtraction resulted in multiple or no polygons.")

    study_area_dict['study_area_cln_poly'] = study_area_subtracted_poly_list[0]
    for class_name, geometry in zip(labels, polygons_to_remove):
        replace_index = (study_area_dict['study_area_rem_poly']['class_name'] == class_name) & \
                        (study_area_dict['study_area_rem_poly']['type'] == type) & \
                        (study_area_dict['study_area_rem_poly']['subtype'] == subtype) # It is a pandas series
        
        if replace_index.sum() > 1:
            logging.error("Multiple polygon matches found for replace_index in study_area_rem_poly.")
            raise ValueError("Multiple polygon matches found for replace_index in study_area_rem_poly.")
        elif replace_index.sum() == 1:
            study_area_dict['study_area_rem_poly'].loc[replace_index, 'geometry'] = geometry
            logging.info(f"Replaced existing polygon for class '{class_name}' in study_area_rem_poly DataFrame.")
        else:
            study_area_dict['study_area_rem_poly'] = pd.concat(
                [
                    study_area_dict['study_area_rem_poly'],
                    pd.DataFrame([{'type': type,
                                   'subtype': subtype,
                                   'class_name': class_name,
                                   'geometry': geometry}]) # Remember that if you use dict, it must be a list of dicts!
                ], ignore_index=True)

        logging.info(f"Updated study_area_rem_poly DataFrame for class '{class_name}'.")
    return study_area_dict

# %% === Main function
def main(source_type: str="land_use", source_subtype: str=None, gui_mode: bool=False, base_dir: str=None):
    if not source_type in KNOWN_OPTIONAL_STATIC_INPUT_TYPES:
        raise ValueError("Invalid source type. Must be one of: " + ", ".join(KNOWN_OPTIONAL_STATIC_INPUT_TYPES))
    
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    study_area_vars = env.load_variable(variable_filename='study_area_vars.pkl')

    if source_subtype:
        rem_filename = f"{source_type}_{source_subtype}_vars.pkl"
    else:
        rem_filename = f"{source_type}_vars.pkl"

    if not os.path.exists(os.path.join(env.folders['variables']['path'], rem_filename)):
        matching_files = [f for f in os.listdir(env.folders['variables']['path']) if f.startswith(f"{source_type}")]
        if len(matching_files) == 0:
            logging.error(f"No existing variable files found for source type '{source_type}' in the variables directory.")
            raise FileNotFoundError(f"No existing variable files found for source type '{source_type}' in the variables directory.")
        elif len(matching_files) == 1:
            rem_filename = matching_files[0]
            logging.info(f"Only one matching file found: {rem_filename}. Using this file.")
        elif len(matching_files) > 1:
            if gui_mode:
                raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
            else:
                rem_filename = select_from_list_prompt(matching_files, "Select the file containing polygons to remove from study area:")[0]
                logging.info(f"Multiple matching files found. Selected file: {rem_filename}")

    rem_poly_vars = env.load_variable(variable_filename=rem_filename)

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        poly_labels_to_remove = select_from_list_prompt(rem_poly_vars['prop_df']['class_name'].to_list(), usr_prompt="Select the classes to remove:", allow_multiple=True)
    
    logging.info(f"Selected classes to remove: {poly_labels_to_remove}")

    study_area_vars = subtract_polygons_from_study_area(
        study_area_dict=study_area_vars,
        rem_poly_df=rem_poly_vars['prop_df'],
        labels=poly_labels_to_remove,
        type=source_type,
        subtype=source_subtype
    )

    env.config['inputs']['study_area'][0]['settings']['source_refined'] = True
    env.save_variable(variable_to_save=study_area_vars, variable_filename='study_area_vars.pkl') # It also updates the environment file
    return study_area_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the study area for the analysis.")
    parser.add_argument('--source_type', type=str, default="land_use", help="Type of the source shapefile (e.g., land_use, soil).")
    parser.add_argument('--source_subtype', type=str, default=None, help="Subtype of the source shapefile (e.g., top, sub).")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    args = parser.parse_args()

    study_area_vars = main(
        source_type=args.source_type,
        source_subtype=args.source_subtype,
        base_dir=args.base_dir,
        gui_mode=args.gui_mode
    )

# %%
