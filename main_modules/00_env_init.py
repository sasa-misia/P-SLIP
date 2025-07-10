#%% Import necessary modules
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


#%% Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

#%% Main function to initialize or load the analysis environment
# This function is responsible for creating or loading the analysis environment based on user input.
def main(case_name=None, gui_mode=False, base_dir=None) -> AnalysisEnvironment:
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
            base_dir = input(f"Enter the base directory for the analysis (or press Enter to use the current directory {os.getcwd()}): ")
            if not base_dir.strip():
                base_dir = os.getcwd()
        
        if case_name is None:
            case_name = input("Specify the analysis name (enter for default [Not Defined - Standalone]): ")
            if not case_name.strip():
                case_name = DEFAULT_CASE_NAME

    os.makedirs(base_dir, exist_ok=True) # Ensure the base directory exists

    # Decide automatically based on base_dir existence
    if os.path.exists(os.path.join(base_dir, ENVIRONMENT_FILENAME)):
        env, _ = get_analysis_environment(base_dir=base_dir)
    else:
        env = create_analysis_environment(base_dir=base_dir, case_name=case_name)

    if not gui_mode:
        logging.info(f"Analysis environment loaded for case: {env.case_name}")
        print(f"Analysis folder structure ready for: {case_name}")

    return env

#%% Command line interface
# This block allows the script to be run from the command line with parameters.
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create or load the folder structure for the analysis.")
    parser.add_argument("--case_name", help="Case study name", default=None)
    parser.add_argument("--base_dir", help="Base directory for the analysis", default=None)
    args = parser.parse_args()

    # Call the main function with the provided arguments
    curr_env = main(case_name=args.case_name, base_dir=args.base_dir, gui_mode=False)