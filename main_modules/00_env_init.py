import argparse
import logging
import os
import sys

# Add the parent directory to the system path (temporarily)
# This allows importing modules from the parent directory (like config and psliptools)
# This is necessary for the script to run correctly when executed directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.analysis_init import create_analysis_environment, get_analysis_environment
from config.default_params import LOG_CONFIG, ENVIRONMENT_FILENAME, DEFAULT_CASE_NAME


# Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# The following method is the main function of the module.
def main(case_name=None, gui_mode=False, base_dir=None):
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
        env = get_analysis_environment(base_dir=base_dir)
    else:
        env = create_analysis_environment(base_dir=base_dir, case_name=case_name)

    if not gui_mode:
        logging.info(f"Analysis environment loaded for case: {env.case_name}")
        print(f"Analysis folder structure ready for: {case_name}")

    return env

# This block allows the script to be run from the command line with parameters.
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create or load the folder structure for the analysis.")
    parser.add_argument("--case_name", help="Case study name", default=None)
    parser.add_argument("--base_dir", help="Base directory for the analysis", default=None)
    args = parser.parse_args()

    # Call the main function with the provided arguments
    curr_env = main(case_name=args.case_name, base_dir=args.base_dir)


# # Check if the raw input files CSV exists and validate paths (TO MOVE TO A UTILITY MODULE LATER)
# if 'internal' in input_files_df.columns:
#     external_mask = ~input_files_df['internal'].astype(bool)
# else:
#     def is_internal(p):
#         abs_p = os.path.abspath(os.path.join(csv_inp_files_dir, p)) if not os.path.isabs(p) else p
#         abs_inp = os.path.abspath(csv_inp_files_dir)
#         return abs_p.startswith(abs_inp)
#     external_mask = ~input_files_df['path'].apply(is_internal)

# if external_mask.any():
#     print("\nSome input files are external to the inputs folder:")
#     for idx, row in input_files_df[external_mask].iterrows():
#         print(f"  - {row['path']}")
#     print(f"\nYou must update the paths of these files manually in {default_params.RAW_INPUT_FILENAME},")
#     print("or you can specify a new path for each file now.")
#     choice = input("Do you want to specify a new path for each external file now? [y/[N]]: ").strip().lower()
#     if choice == "y":
#         for idx in input_files_df[external_mask].index:
#             old_path = input_files_df.at[idx, 'path']
#             new_path = input(f"Enter new absolute path for file '{old_path}': ").strip()
#             if new_path:
#                 input_files_df.at[idx, 'path'] = new_path
#         input_files_df.to_csv(csv_inp_files_path, index=False)
#         print(f"{default_params.RAW_INPUT_FILENAME} updated with new paths.")
#     else:
#         print(f"Please update {default_params.RAW_INPUT_FILENAME} manually before proceeding.")