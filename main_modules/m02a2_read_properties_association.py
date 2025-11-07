# %% === Import necessary modules
import os
import sys
import pandas as pd
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    KNOWN_OPTIONAL_STATIC_INPUT_TYPES,
    STANDARD_CLASSES_FILENAME,
    PARAMETER_CLASSES_FILENAME,
    SUPPORTED_FILE_TYPES
)

from psliptools.utilities import (
    select_file_prompt,
    select_from_list_prompt
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger, log_and_error
logger = setup_logger(__name__)
logger.info("=== Associate properties to polygons ===")

# %% === Helper functions
def class_association(
        prop_df: pd.DataFrame, 
        association_df: pd.DataFrame,
        prop_class_id_column: str, 
        association_class_id_column: str, 
        reference_classes_df: pd.DataFrame,
        ) -> tuple[pd.DataFrame, bool]:
        """Helper function to process class associations"""
        processed = False
        if association_df[association_class_id_column].any():
            only_associated_df = association_df[~association_df[association_class_id_column].isna()]
            for _, row in only_associated_df.iterrows():
                idx_to_replace = prop_df['class_name'] == row['class_name']
                if idx_to_replace.sum() > 1:
                    log_and_error(f"Multiple rows with the same class name: {row['class_name']}", ValueError, logger)
                elif idx_to_replace.sum() == 0:
                    log_and_error(f"No rows with the class name: {row['class_name']}", ValueError, logger)
                
                if not row[association_class_id_column] in reference_classes_df['class_id'].values:
                    log_and_error(f"Invalid {association_class_id_column}: {row[association_class_id_column]}. Possible values: {list(reference_classes_df['class_id'].values)}", ValueError, logger)
                
                prop_df.loc[idx_to_replace, prop_class_id_column] = row[association_class_id_column]

            processed = True
            
        return prop_df, processed

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        source_type: str="land_use", 
        source_subtype: str=None, 
        standard_classes_filepath: str=None, 
        parameter_classes_filepath: str=None
    ) -> dict[str, object]: 
    if not source_type in KNOWN_OPTIONAL_STATIC_INPUT_TYPES:
        log_and_error("Invalid source type. Must be one of: " + ", ".join(KNOWN_OPTIONAL_STATIC_INPUT_TYPES), ValueError, logger)
    
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    possible_subtypes = [x['settings']['source_subtype'] for x in env.config['inputs'][source_type]]

    if gui_mode:
        log_and_error("GUI mode is not supported in this script yet. Please run the script without GUI mode.", NotImplementedError, logger)
    else:
        if not standard_classes_filepath:
            print("\n=== Association file for standard classes ===")
            standard_classes_filepath = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the standard classes association file (default: [{STANDARD_CLASSES_FILENAME}]): ",
                src_ext=SUPPORTED_FILE_TYPES['table'],
                default_file=STANDARD_CLASSES_FILENAME
            )

        if not parameter_classes_filepath:
            print("\n=== Association file for parameter classes ===")
            parameter_classes_filepath = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the parameter classes association file (default: [{PARAMETER_CLASSES_FILENAME}]): ",
                src_ext=SUPPORTED_FILE_TYPES['table'],
                default_file=PARAMETER_CLASSES_FILENAME
            )
        
        if not source_subtype and len(env.config['inputs'][source_type]) > 1:
            source_subtype = select_from_list_prompt(possible_subtypes, "Select a subtype")[0]
    
    standard_classes_df = pd.read_csv(standard_classes_filepath)
    parameter_classes_df = pd.read_csv(parameter_classes_filepath)

    if len(env.config['inputs'][source_type]) == 1:
        curr_idx = 0
    elif len(env.config['inputs'][source_type]) > 1:
        curr_idx = possible_subtypes.index(source_subtype)
    
    association_df = pd.read_csv(os.path.join(env.folders['user_control']['path'], env.config['inputs'][source_type][curr_idx]['settings']['association_filename']))

    if source_subtype:
        rel_filename = f"{source_type}_{source_subtype}"
    else:
        rel_filename = f"{source_type}"

    prop_vars = env.load_variable(variable_filename=f'{rel_filename}_vars.pkl')
    prop_df = prop_vars['prop_df']

    prop_df, standard_exists = class_association(prop_df, association_df, 'standard_class', 'standard_class', standard_classes_df)
    prop_df, parameter_exists = class_association(prop_df, association_df, 'parameter_class', 'parameter_class', parameter_classes_df)

    prop_vars['prop_df'] = prop_df

    if standard_exists:
        env.config['inputs'][source_type][curr_idx]['settings']['standard_filename'] = os.path.basename(standard_classes_filepath)
    
    if parameter_exists:
        env.config['inputs'][source_type][curr_idx]['settings']['parameter_filename'] = os.path.basename(parameter_classes_filepath)
        
    env.save_variable(variable_to_save=prop_vars, variable_filename=f"{rel_filename}_vars.pkl")

    logger.info(f"Associations processed for {source_type}.")

    return prop_vars


# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Associate main properties with parameters.")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for the analysis")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    parser.add_argument("--source_type", type=str, default="land_use", help="Source type (e.g., " + ", ".join(KNOWN_OPTIONAL_STATIC_INPUT_TYPES) + ")")
    parser.add_argument("--source_subtype", type=str, default=None, help="Source subtype (e.g., top, sub) (optional)")
    parser.add_argument("--standard_classes_filepath", type=str, default=None, help="Path to the standard classes association file")
    parser.add_argument("--parameter_classes_filepath", type=str, default=None, help="Path to the parameter classes association file")
    args = parser.parse_args()

    prop_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        source_type=args.source_type, 
        source_subtype=args.source_subtype,
        standard_classes_filepath=args.standard_classes_filepath,
        parameter_classes_filepath=args.parameter_classes_filepath
    )