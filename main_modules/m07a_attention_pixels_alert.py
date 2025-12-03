# %% === Import necessary modules
import argparse
import numpy as np
import pandas as pd

# Importing necessary modules from main_modules
from m00a_env_init import get_or_create_analysis_environment, setup_logger, log_and_warning, log_and_error, memory_report, obtain_config_idx_and_rel_filename
logger = setup_logger(__name__)
logger.info("=== Attention pixels analysis and alert ===")

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    REFERENCE_POINTS_FILENAME,
    SUPPORTED_FILE_TYPES,
    KNOWN_DYNAMIC_INPUT_TYPES,
    DYNAMIC_SUBFOLDERS
)

# # Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

from psliptools.utilities import (
    select_from_list_prompt,
    select_file_prompt,
    read_generic_csv
)

# from psliptools.geometries import (
# )

# from psliptools.scattered import (
# )

# %% === Helper functions
LANDSLIDE_PATHS_FILENAME = "landslide_paths_vars.pkl"
POSSIBLE_TRIGGER_MODES = ['rainfall-threshold', 'safety-factor', 'machine-learning']

def get_top_k_paths(
        paths_df: pd.DataFrame,
        dtm: int,
        point_2d: tuple[int, int] | list[int] | np.ndarray,
        k: int = 3
    ) -> list[str]:
    """
    Return top-k path_ids (highest path_realism_score) passing through point_2d in given dtm.
    
    Args:
        paths_df (pd.DataFrame): DataFrame containing landslide paths data.
        dtm (int): DTM value.
        point_2d (tuple[int, int] | list[int] | np.ndarray): 2D index of the point.
        k (int, optional): Number of top paths to return (default is 3).

    Returns:
        list[str]: List of top-k path_ids.
    """
    if not isinstance(paths_df, pd.DataFrame):
        log_and_error("paths_df must be a pandas DataFrame.", ValueError, logger)
    if not isinstance(dtm, int):
        log_and_error("dtm must be an integer.", ValueError, logger)
    if not isinstance(point_2d, (tuple, list, np.ndarray)):
        log_and_error("point_2d must be a tuple, list, or numpy array.", ValueError, logger)
    if k <= 0:
        log_and_error("k must be positive.", ValueError, logger)

    point_2d = np.atleast_1d(point_2d)
    if point_2d.size != 2:
        log_and_error("point_2d must be a tuple, list, or numpy array of length 2.", ValueError, logger)
    
    curr_df = paths_df[paths_df['path_dtm'] == dtm]

    candidates = []
    for _, row in curr_df.iterrows():
        if (row['path_2D_idx'] == point_2d).all(axis=1).any():
            candidates.append((row['path_realism_score'], row['path_id']))
    candidates.sort(reverse=True, key=lambda x: x[0])

    top_paths = [pid for _, pid in candidates[:k]]

    return top_paths + [None] * (k - len(top_paths))

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        trigger_mode: str='rainfall-threshold', # or 'safety-factor' or 'machine-learning'
        alert_threshold: float=None # value and measure unit depending on the trigger mode
    ) -> dict[str, object]:
    """Main function to analyze and alert on attention pixels."""

    # Input validation
    if trigger_mode not in POSSIBLE_TRIGGER_MODES:
        log_and_error(f"Invalid trigger_mode: {trigger_mode}. Must be one of {POSSIBLE_TRIGGER_MODES}.", ValueError, logger)

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    landslide_paths_vars = env.load_variable(variable_filename=LANDSLIDE_PATHS_FILENAME)

    landslide_paths_df = landslide_paths_vars['paths_df']

    initial_dtms = np.sort(landslide_paths_df['path_dtm'].unique())
    dtm_unique_points: list[np.ndarray] = []
    for curr_dtm in initial_dtms:
        curr_dtm_paths_df = landslide_paths_df[landslide_paths_df['path_dtm'] == curr_dtm]
        
        all_point_arrays = np.vstack(curr_dtm_paths_df['path_2D_idx'].values)  # (N, 2)
        unique_pts = np.unique(all_point_arrays, axis=0)
        dtm_unique_points.append(unique_pts)
    
    attention_pixels_df = pd.DataFrame({
        'dtm': initial_dtms,
        '2D_idx': dtm_unique_points
    })

    if trigger_mode == 'rainfall-threshold':
        if gui_mode:
            source_type = 'rain'
            source_subtype = select_from_list_prompt(
                obj_list=DYNAMIC_SUBFOLDERS,
                usr_prompt='Select the source subtype: ',
                allow_multiple=False
            )[0]

            env, idx_config, rel_filename = obtain_config_idx_and_rel_filename(env, source_type, source_subtype)

            source_mode = env.config['inputs'][source_type][idx_config]['settings']['source_mode']
            if not source_mode == 'station':
                log_and_error("Invalid source mode. Must be 'station'", ValueError, logger)

            alert_metric = select_from_list_prompt(
                obj_list=['cumulative_rain'],
                usr_prompt='Select the alert metric to use for triggering: ',
                allow_multiple=False
            )

            # TODO: Implement the rest of the code
        else:
            log_and_error("GUI mode not implemented yet.", NotImplementedError, logger)
    
    elif trigger_mode == 'safety-factor':
        log_and_error(f"Trigger mode {trigger_mode} is not implemented yet.", NotImplementedError, logger)

    elif trigger_mode == 'machine-learning':
        log_and_error(f"Trigger mode {trigger_mode} is not implemented yet.", NotImplementedError, logger)
    
    else:
        log_and_error(f"Trigger mode not recognized or not implemented: {trigger_mode}.", ValueError, logger)
    
    # TODO: Implement the rest of the code
    out_vars = {}

    return out_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and alert on attention pixels.")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    
    args = parser.parse_args()
    
    out_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode
    )