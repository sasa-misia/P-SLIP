# %% === Import necessary modules
import argparse
import logging
import os
import sys

# Add the parent directory to the system path (temporarily)
# This allows importing modules from the parent directory (like config and psliptools)
# This is necessary for the script to run correctly when executed directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.analysis_init import create_analysis_environment, get_analysis_environment, AnalysisEnvironment
from config.default_params import LOG_CONFIG, ENVIRONMENT_FILENAME, DEFAULT_CASE_NAME


# %% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# %% === Main function to initialize or load the analysis environment
# This function is responsible for creating or loading the analysis environment based on user input.
def get_or_create_analysis_environment(
        case_name=None,
        base_dir=None, 
        allow_creation=True,
        env_filename=None,
        gui_mode=False
    ) -> AnalysisEnvironment:
    """
    Main function for initializing or loading the analysis environment.

    Args:
        case_name: Case study name. If None, a default name is used.
        gui_mode: If true, the function was called from a GUI.
        base_dir: Base directory for the analysis. If None, the current directory is used.

    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """

    # Prompt for case name and base directory if not provided
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        if base_dir is None:
            base_dir = input(f"Enter the base directory for the analysis (or press Enter to use the current directory [{os.getcwd()}]): ").strip(' "')
            if not base_dir:
                base_dir = os.getcwd()

    os.makedirs(base_dir, exist_ok=True) # Ensure the base directory exists

    if env_filename:
        if not isinstance(env_filename, str):
            raise ValueError("env_file_path must be a string representing the path to the environment file.")
        if not env_filename.endswith('.json'):
            raise ValueError("env_file_path must be a JSON file.")
        if os.path.isabs(env_filename):
            env_filename = os.path.basename(env_filename)
        CURR_ENV_FILENAME = env_filename
    else:
        CURR_ENV_FILENAME = ENVIRONMENT_FILENAME

    # Decide automatically based on base_dir existence
    if os.path.exists(os.path.join(base_dir, CURR_ENV_FILENAME)):
        env = get_analysis_environment(base_dir=base_dir)
    else:
        if allow_creation:
            if case_name is None:
                if gui_mode:
                    raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
                else:
                    case_name = input("Specify the analysis name (enter for default [Not Defined - Standalone]): ").strip()
                    if not case_name:
                        case_name = DEFAULT_CASE_NAME
            env = create_analysis_environment(base_dir=base_dir, case_name=case_name)
        else:
            raise FileNotFoundError(f"Analysis environment file '{CURR_ENV_FILENAME}' not found in {base_dir}. "
                                     "Please create the environment first or set allow_creation=True.")

    logging.info(f"Analysis environment loaded for case: {env.case_name}")
    return env

# %% Utuility functions
def obtain_config_idx_and_rel_filename(
        env: AnalysisEnvironment, 
        source_type: str, 
        source_subtype: str=None
    ) -> tuple[AnalysisEnvironment, int, str]:
    """
    Utility function to obtain the config index and relative filename based on the source type and subtype.

    Args:
        env (AnalysisEnvironment): The analysis environment object.
        source_type (str): The source type.
        source_subtype (str, optional): The source subtype. Defaults to None.

    Returns:
        tuple[AnalysisEnvironment, int, str]: The modified analysis environment, the config index, and the possible relative filename to use for output.
    """
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

# %% === Command line interface
# This block allows the script to be run from the command line with parameters.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or load the folder structure for the analysis.")
    parser.add_argument("--case_name", type=str, default=None, help="Case study name")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for the analysis")
    parser.add_argument("--allow_creation", action="store_true", help="Allow creation of the environment if it doesn't exist")
    parser.add_argument("--env_filename", type=str, default=None, help="Environment filename")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode (not implemented yet)")
    args = parser.parse_args()

    # Call the main function with the provided arguments
    curr_env = get_or_create_analysis_environment(
        case_name=args.case_name, 
        base_dir=args.base_dir,
        allow_creation=args.allow_creation,
        env_filename=args.env_filename,
        gui_mode=args.gui_mode
    )