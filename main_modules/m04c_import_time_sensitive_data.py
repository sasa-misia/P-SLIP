# %% === Import necessary modules
import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    KNOWN_DYNAMIC_INPUT_TYPES,
    DYNAMIC_SUBFOLDERS
)

# Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

from psliptools.utilities import (
    select_dir_prompt,
    select_files_in_folder_prompt,
    select_file_prompt
)

from psliptools.scattered import (
    load_time_sensitive_data_from_csv
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, obtain_config_idx_and_rel_filename, setup_logger
logger = setup_logger()
logger.info("=== Importing time-sensitive data ===")

# %% === Methods to import time sensitive data as rainfall and temperature
SOURCE_MODES = ['station', 'satellite']

# %% === Main function
def main(
        gui_mode: bool=False, 
        base_dir: str=None,
        source_mode: str="station",
        source_type: str="rain", 
        source_subtype: str="recordings",
        delta_time_hours: float=1,
        aggregation_mode: str=["sum"],
        last_date: pd.Timestamp=None
    ) -> None:
    """Main function to import time sensitive data"""
    if not source_mode in SOURCE_MODES:
        raise ValueError("Invalid source mode. Must be one of: " + ", ".join(SOURCE_MODES))
    if not source_type in KNOWN_DYNAMIC_INPUT_TYPES:
        raise ValueError("Invalid source type. Must be one of: " + ", ".join(KNOWN_DYNAMIC_INPUT_TYPES))
    if not source_subtype in DYNAMIC_SUBFOLDERS:
        raise ValueError("Invalid source subtype. Must be one of: " + ", ".join(DYNAMIC_SUBFOLDERS[source_type]))
    
    if source_mode == 'station':
        allowed_extensions = ['.csv'] # Other extensions must be implemented
    elif source_mode == 'satellite':
        allowed_extensions = ['.nc'] # Other extensions must be implemented
    
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    env, idx_config, rel_filename = obtain_config_idx_and_rel_filename(env, source_type, source_subtype)

    study_area_cln_poly = env.load_variable(variable_filename='study_area_vars.pkl')['study_area_cln_poly']

    logger.info(f"Importing time-sensitive data [{source_type}] from [{source_subtype}] in [{source_mode}] mode...")
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Directory selection ===")
        data_fold = select_dir_prompt(
            default_dir=env.folders['inputs'][source_type][source_subtype]['path'], 
            content_type=source_type
        )
        
        print("\n=== Data files selection ===")
        data_paths = select_files_in_folder_prompt(
            base_dir=data_fold, 
            src_ext=allowed_extensions, 
            allow_multiple=True, 
            usr_prompt="Number, name or full path of the data files (ex. data.csv): "
        )

        gauge_info_path = None
        if source_mode == 'station':
            print("\n=== Gauge info file selection ===")
            gauge_info_path = select_file_prompt(
                base_dir=env.folders['inputs'][source_type][source_subtype]['path'], 
                usr_prompt="Name or full path of the gauge info file (ex. gauge_info.csv): ", 
                src_ext=allowed_extensions
            )

    if gauge_info_path in data_paths:
        data_paths.remove(gauge_info_path)
    
    if source_type == 'rain':
        fill_method = 'zero'
    elif source_type == 'temperature':
        fill_method = 'linear'
    else:
        fill_method = None

    data_list = []
    for data_pth in data_paths:
        data_list.append(
            load_time_sensitive_data_from_csv(
                file_path=data_pth,
                value_names=None,
                fill_method=fill_method,
                round_datetime=True,
                delta_time_hours=delta_time_hours,
                aggregation_method=aggregation_mode,
                last_date=last_date
            )
        )
    

    
    # TODO: Add code to import time-sensitive data as rainfall and temperature

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import time-sensitive data (rainfall and temperature)")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    
    args = parser.parse_args()
    
    main(gui_mode=args.gui_mode, base_dir=args.base_dir)