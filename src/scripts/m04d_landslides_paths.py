# %% === Import necessary modules
import os
import argparse
import numpy as np
import pandas as pd
from functools import lru_cache
from collections import deque

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Importing necessary modules from main_modules
from m00a_env_init import get_or_create_analysis_environment, setup_logger, log_and_warning, log_and_error, memory_report, get_hardware_info
from m05a_reference_points_info import convert_abg_and_ref_points_to_prj
logger = setup_logger(__name__)
logger.info("=== Landslide paths creation ===")

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    SUPPORTED_FILE_TYPES,
    REFERENCE_POINTS_FILENAME
)

# Importing necessary modules from psliptools
from psliptools.rasters import (
    get_2d_idx_from_1d_idx,
    get_d8_neighbors_slope,
    get_point_gradients,
    get_closest_1d_pixel_idx,
    get_closest_grid_and_1d_pixel_idx,
    convert_coords
)

from psliptools.utilities import (
    select_file_prompt,
    read_generic_csv
)

# %% === Helper functions and global variables
current_tot_ram = get_hardware_info()['ram']
MAXIMUM_COMPATIBLE_PATHS = int(1700000 / 19.76 * current_tot_ram) if current_tot_ram > 0 else 1700000 # Tested on 19.76 (~24 - 4 GB for iGPU) GB RAM mini pc and handled up to 1700000 paths
VARIABLE_FILENAME = "landslide_paths_vars.pkl"
POSSIBLE_METHODS = ['gradient', 'd8-flow', 'slope-compatible']
POSSIBLE_FLOW_SENSES = ['upstream', 'downstream']
POSSIBLE_STARTING_SOURCES = ['reference_points', 'landslides_dataset', 'potential_landslides']
STOP_REASONS = {
    'edge': 'edge_reached', 
    'loop': 'loop_detected', 
    'invalid': 'invalid_steps_tolerance_reached', 
    'opposite': 'greater_than_tolerance_opposite_flow', 
    'max_steps': 'max_runout', 
    'max_paths': 'partial_because_max_paths_exceeded',
    'low_realism': 'low_partial_realism_score',
    'complete': 'completed'
}

def get_starting_points(
        env: AnalysisEnvironment,
        starting_source: str,
        file_path: str=None,
        id_column: str="id",
        x_column: str="lon",
        y_column: str="lat"
    ) -> pd.DataFrame:
    """Load starting points based on starting_source."""
    if starting_source == 'reference_points':
        raw_df = read_generic_csv(file_path)
    elif starting_source == 'landslides_dataset':
        raw_df = env.load_variable(variable_filename='landslides_vars.pkl')
        log_and_error("landslides_dataset starting_source is not implemented yet.", NotImplementedError, logger)
    elif starting_source == 'potential_landslides':
        log_and_error("potential_landslides starting_source is not implemented yet.", NotImplementedError, logger)
    else:
        log_and_error(f"Unknown starting_source: {starting_source}", ValueError, logger)
    
    starting_points_df = raw_df[[id_column, x_column, y_column]].copy().reset_index(drop=True)
    
    return starting_points_df

def generate_steepsets_path(
        start_idx: tuple[int, int], 
        x_grid: np.ndarray, 
        y_grid: np.ndarray, 
        z_grid: np.ndarray, 
        method: str, 
        flow: str, 
        step_size: int, 
        max_steps: int, 
        min_slope: float, 
        invalid_steps_tolerance: int
    ) -> tuple[np.ndarray, np.ndarray, int, str, np.ndarray, np.ndarray]:
    """Generate a single path from start point."""
    path_coord_list = [( # x, y, z
        x_grid[start_idx[0], start_idx[1]],
        y_grid[start_idx[0], start_idx[1]],
        z_grid[start_idx[0], start_idx[1]]
    )]
    path_2D_idx_list = [start_idx]
    steps_count = 0
    invalid_steps = 0
    stop_reason = None
    curr_row, curr_col = start_idx
    step_deviation_list = []
    step_validity_list = []

    dx = np.abs(np.mean(x_grid[curr_row, 1:] - x_grid[curr_row, :-1]))
    dy = np.abs(np.mean(y_grid[1:, curr_col] - y_grid[:-1, curr_col]))

    curr_x, curr_y = path_coord_list[0][0].copy(), path_coord_list[0][1].copy()
    
    while steps_count < max_steps:
        # --- Determine next position
        if method == 'gradient':
            # Compute gradient at current position
            grad_x, grad_y = get_point_gradients(
                row=curr_row, 
                col=curr_col, 
                z_grid=z_grid, 
                x_grid=x_grid, 
                y_grid=y_grid,
                search_size=step_size
            )
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) # Always positive
            
            # Determine direction: negative gradient for descent, positive for ascent
            if flow == 'downstream':
                direction_x = -grad_x
                direction_y = -grad_y
            elif flow == 'upstream':
                direction_x = grad_x
                direction_y = grad_y
            else:
                log_and_error(f"Unknown flow type: {flow}", ValueError, logger)
            
            # Update position with learning rate (step_size)
            curr_x = curr_x + 10 * dx * step_size * direction_x # 10 is just a learning rate to speed up the process
            curr_y = curr_y + 10 * dy * step_size * direction_y # 10 is just a learning rate to speed up the process
            
            # Check if new position is within grid bounds
            x_min, x_max = np.min(x_grid), np.max(x_grid)
            y_min, y_max = np.min(y_grid), np.max(y_grid)
            if not (x_min <= curr_x <= x_max and y_min <= curr_y <= y_max):
                stop_reason = STOP_REASONS['edge']
                break

            # Obtain a subset of the grid that contains the new position
            x_cells_margin = np.ceil(abs(curr_x - x_grid[curr_row, curr_col]) / dx).astype(int) + 1
            y_cells_margin = np.ceil(abs(curr_y - y_grid[curr_row, curr_col]) / dy).astype(int) + 1
            sub_row_start = curr_row - y_cells_margin
            sub_row_end = curr_row + y_cells_margin
            sub_col_start = curr_col - x_cells_margin
            sub_col_end = curr_col + x_cells_margin
            sub_x_grid = x_grid[sub_row_start : sub_row_end, sub_col_start : sub_col_end]
            sub_y_grid = y_grid[sub_row_start : sub_row_end, sub_col_start : sub_col_end]
            
            # Find nearest grid index to new position
            next_1d_idx, _ = get_closest_1d_pixel_idx(
                x=curr_x, 
                y=curr_y, 
                x_grid=sub_x_grid, 
                y_grid=sub_y_grid, 
                par_grid_ref=(
                    sub_row_start, 
                    sub_col_start, 
                    x_grid.shape[0], 
                    x_grid.shape[1]
                )
            )
            next_row, next_col = get_2d_idx_from_1d_idx(indices=next_1d_idx, shape=z_grid.shape)

            # Calculate deviation: angle difference from max slope direction
            max_slope_direction = np.arctan2(direction_y, direction_x)
            actual_direction = np.arctan2(curr_y - y_grid[curr_row, curr_col], curr_x - x_grid[curr_row, curr_col])
            angle_diff = np.abs(np.arctan2(np.sin(max_slope_direction - actual_direction), np.cos(max_slope_direction - actual_direction)))
            step_deviation_list.append(angle_diff / np.deg2rad(45)) # Theoretically, it can be at max 4.0 (180/45), but following the gradient and snapping, it could be at max 1.0 in this case

            if grad_magnitude < min_slope:  # Use gradient magnitude as slope proxy. Just < and not <= because the min slope is included in valid steps
                invalid_steps += 1
                step_validity_list.append(False)
            else:
                invalid_steps = 0
                step_validity_list.append(True)
        elif method == 'd8-flow':
            # D8: Choose direction with steepest drop/rise
            slopes_df = get_d8_neighbors_slope(
                row=curr_row, 
                col=curr_col, 
                z_grid=z_grid,
                x_grid=x_grid,
                y_grid=y_grid,
                search_size=step_size,
                output_format='pandas'
            )

            neighbors = slopes_df[['row_end', 'col_end']].values
            slopes = slopes_df['slope'].values # Positive means uphill

            if np.isnan(slopes).any():
                stop_reason = STOP_REASONS['edge']
                break

            if flow == 'downstream':
                best_idx = np.argmin(slopes)  # Steepest descent
            elif flow == 'upstream':
                best_idx = np.argmax(slopes)  # Steepest ascent
            else:
                log_and_error(f"Unknown flow type: {flow}", ValueError, logger)
            
            next_row, next_col = neighbors[best_idx]
            slope = slopes[best_idx]

            step_deviation_list.append(0.0) # For d8-flow, deviation is 0 since we always choose the max / min slope

            if abs(slope) < min_slope: # Just < and not <= because the min slope is included in valid steps
                invalid_steps += 1
                step_validity_list.append(False)
            else:
                if flow == 'downstream' and slope < 0 or flow == 'upstream' and slope > 0:
                    invalid_steps = 0
                    step_validity_list.append(True)
                else: # Sometimes it can happen that the argmin or argmax finds the opposite direction and exceeds the tolerance
                    stop_reason = STOP_REASONS['opposite']
                    break
        else:
            log_and_error(f"Unknown method: {method}", ValueError, logger)
        
        # --- Check stopping criteria
        if (next_row, next_col) in path_2D_idx_list: # Check for loop (bouncing back and forth)
            stop_reason = STOP_REASONS['loop']
            break
        
        if invalid_steps > invalid_steps_tolerance: # Check invalid steps tolerance, just > because this invalid point has not been added yet
            stop_reason = STOP_REASONS['invalid']
            break
        
        # --- Move to next point
        curr_row, curr_col = next_row, next_col
        path_coord_list.append((x_grid[curr_row, curr_col], y_grid[curr_row, curr_col], z_grid[curr_row, curr_col]))
        path_2D_idx_list.append((curr_row, curr_col))
        steps_count += 1
    
    # --- Check stopping criteria
    if steps_count >= max_steps: # Check max runout
        stop_reason = STOP_REASONS['max_steps']

    if stop_reason is None: # Check if path completed
        stop_reason = STOP_REASONS['complete']

    # --- Convert lists to arrays
    path_coords = np.array(path_coord_list)
    path_2D_idx = np.array(path_2D_idx_list)
    path_step_deviation = np.array(step_deviation_list)
    path_step_validity = np.array(step_validity_list)
    
    return path_coords, path_2D_idx, steps_count, stop_reason, path_step_deviation, path_step_validity

def generate_slope_compatible_paths(
        start_idx: tuple[int, int],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        flow: str,
        step_size: int,
        max_steps: int,
        min_slope: float,
        invalid_steps_tolerance: int,
        maximum_paths: int=MAXIMUM_COMPATIBLE_PATHS,
        allow_low_slope_opposite_flow: bool=False,
        min_partial_realism_score: float=None,
        verbose: bool=False
    ) -> list[tuple[np.ndarray, np.ndarray, int, str, np.ndarray, np.ndarray]]:
    """
    Generate paths that are compatible with the slope of the terrain.

    Args:
        start_idx (tuple[int, int]): The starting index of the path.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        z_grid (np.ndarray): The z coordinates of the grid.
        flow (str): The flow direction of the path. Can be 'downstream' or 'upstream'.
        step_size (int): The step size of the path.
        max_steps (int): The maximum number of steps in the path.
        min_slope (float): The minimum slope of the path.
        invalid_steps_tolerance (int): The maximum number of invalid steps in the path.
        maximum_paths (int, optional): The maximum number of paths to generate. Defaults to 50000.
        allow_low_slope_opposite_flow (bool, optional): Whether to allow low slope in opposite flow direction. Defaults to False.
        min_partial_realism_score (float, optional): Minimum partial realism score to continue path exploration. If None, no early stopping based on score. Defaults to None.
        verbose (bool, optional): Whether to print progress updates. Defaults to False.

    Returns:
        list[tuple[np.ndarray, np.ndarray, int, str, np.ndarray, np.ndarray]]: A list of tuples containing the path coordinates, path 2D index, number of steps, stop reason, deviation array, valid steps array.
    """

    @lru_cache(maxsize=128)
    def _cached_neighbors(row: int, col: int, step_size: int):
        """Cache wrapper for get_d8_neighbors_slope."""
        return get_d8_neighbors_slope(
            row=row,
            col=col,
            z_grid=z_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            search_size=step_size,
            output_format='pandas'
        )
    
    paths = []
    start_coords = (
        x_grid[start_idx[0], start_idx[1]],
        y_grid[start_idx[0], start_idx[1]],
        z_grid[start_idx[0], start_idx[1]]
    )

    # Use deque for more efficient stack operations (O(1) append/pop)
    stack = deque([([start_coords], [start_idx], 0, 0, [], [])])  # (coords, idx, steps, invalid_steps_count, deviation_list, valid_steps_list)

    while stack:
        curr_coords, curr_idx, steps, invalid, curr_deviation, curr_valid = stack.pop()

        # Early check for max paths to avoid unnecessary computations
        if len(paths) + len(stack) >= maximum_paths:
            log_and_warning(f"Maximum paths limit exceeded! Completed paths: {len(paths)}, remaining paths to explore: {len(stack)}. Stopping exploration...", stacklevel=2, logger=logger)
            for coord, idx, steps, _, dev, valid in stack:
                paths.append((np.array(coord), np.array(idx), steps, STOP_REASONS['max_paths'], np.array(dev), np.array(valid)))
            if verbose:
                print(
                    f"Exploration was stopped because the maximum number of paths was exceeded. Completed paths: {len(paths) - len(stack)}, remaining paths to explore: {len(stack)}."
                )
            break

        # --- early termination: max runout ----------
        if steps >= max_steps:
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, STOP_REASONS['max_steps'], np.array(curr_deviation), np.array(curr_valid)))
            if verbose:
                print(f"Path n. {len(paths)} has reached max runout; stopped at step: {steps} ...")
            continue
        # --------------------------------------------

        slopes_df = _cached_neighbors(curr_idx[-1][0], curr_idx[-1][1], step_size)
        neighbors = slopes_df[['row_end', 'col_end']].values
        slopes = slopes_df['slope'].values

        # --- early termination: edge reached --------
        if np.isnan(slopes).any():
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, STOP_REASONS['edge'], np.array(curr_deviation), np.array(curr_valid)))
            if verbose:
                print(f"Path n. {len(paths)} has reached an edge; stopped at step: {steps} ...")
            continue
        # --------------------------------------------

        # Vectorized slope masks for efficiency
        if flow == 'downstream':
            valid_mask = slopes <= -min_slope
            if allow_low_slope_opposite_flow:
                low_mask = np.abs(slopes) < min_slope
            else:
                low_mask = (slopes > -min_slope) & (slopes <= 0)
        elif flow == 'upstream':
            valid_mask = slopes >= min_slope
            if allow_low_slope_opposite_flow:
                low_mask = np.abs(slopes) < min_slope
            else:
                low_mask = (slopes < min_slope) & (slopes >= 0)
        else:
            log_and_error(f"Invalid flow sense: {flow}", ValueError, logger)

        compatible_mask = valid_mask | low_mask # NOTE: not compatible_mask represents neighbors with the opposite flow with sloped terrain (greater than min_slope and in opposite flow direction)

        # --- early termination: all neighbors with opposite flow ----------
        if not compatible_mask.any():
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, STOP_REASONS['opposite'], np.array(curr_deviation), np.array(curr_valid)))
            if verbose:
                print(f"Path n. {len(paths)} has reached all neighbors with opposite flow; stopped at step: {steps} ...")
            continue
        # ------------------------------------------------------------------

        # --- early termination: invalid steps tolerance exceeded with all compatible neighbors ----------
        if not valid_mask.any() and low_mask.any() and (invalid == invalid_steps_tolerance): # All compatible neighbors would be other invalid points, thus exceeding tolerance
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, STOP_REASONS['invalid'], np.array(curr_deviation), np.array(curr_valid)))
            if verbose:
                print(f"path n. {len(paths)} has reached all compatible neighbors with consecutive invalid steps tolerance; stopped at step: {steps} ...")
            continue
        # ------------------------------------------------------------------------------------------------

        # --- early termination: low partial realism score ----------
        if min_partial_realism_score is not None and steps >= max(4, invalid_steps_tolerance):
            # TODO: Check and test in different senarios!
            # Calculate partial realism score (simplified version without path_length and stop_penalty)
            valid_steps_ratio = np.mean(curr_valid) if len(curr_valid) > 0 else 0
            avg_deviation = np.mean(curr_deviation) if len(curr_deviation) > 0 else 0
            
            # Calculate average slope from current path
            if len(curr_coords) > 1:
                curr_coords_array = np.array(curr_coords)
                diff_xy = np.diff(curr_coords_array[:, 0:2], axis=0)
                diff_z = np.diff(curr_coords_array[:, 2], axis=0)
                path_step_plane_length = np.sqrt(np.sum(diff_xy ** 2, axis=1))
                path_step_slope_degrees = np.rad2deg(np.arctan2(diff_z, path_step_plane_length))
                avg_slope = np.mean(path_step_slope_degrees)
            else:
                avg_slope = 0
            
            # Slope penalty: penalize if average slope is below min_slope (converted back to degrees)
            min_slope_degrees = np.rad2deg(np.arctan(min_slope))
            average_slope_penalty = min(60, max(0, (min_slope_degrees - abs(avg_slope)) * 5))
            
            # Partial realism score (max 70 instead of 100, as we don't have path_length bonus and stop_penalty)
            partial_realism_score = max(0, (valid_steps_ratio * 70) - (avg_deviation * 5) - average_slope_penalty)
            
            if partial_realism_score < min_partial_realism_score:
                paths.append((np.array(curr_coords), np.array(curr_idx), steps, STOP_REASONS['low_realism'], np.array(curr_deviation), np.array(curr_valid)))
                if verbose:
                    print(f"Path n. {len(paths)} has low partial realism score ({partial_realism_score:.1f} < {min_partial_realism_score}); stopped at step: {steps} ...")
                continue
        # -----------------------------------------------------------

        # Directions: 0=E, 45=SE, 90=S, 135=SW, 180=W, 225=NW, 270=N, 315=NE
        if (slopes_df.index == ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']).all():
            directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        else:
            log_and_error(f"Invalid slope directions: {slopes_df.index}", ValueError, logger)

        curr_row_col_set = set(curr_idx)  # Faster lookup for loops
        for i in np.where(compatible_mask)[0]: # Only iterate over compatible neighbors
            nr, nc = neighbors[i]
            if (nr, nc) in curr_row_col_set:
                continue # Skip because this neighbor has already been visited and it would be a loop

            is_valid = valid_mask[i]
            is_low = low_mask[i]
            new_invalid = invalid + 1 if is_low else 0
            if new_invalid > invalid_steps_tolerance:
                continue # Skip because this branch would exceed tolerance of invalid consecutive steps

            # Calculate deviation: angular distance from max slope direction
            if flow == 'downstream':
                max_slope_idx = np.argmin(slopes)
            elif flow == 'upstream':
                max_slope_idx = np.argmax(slopes)
            
            angle_i = directions[i]
            angle_max = directions[max_slope_idx]
            angle_diff = min(np.abs(angle_i - angle_max), 360 - np.abs(angle_i - angle_max))
            new_deviation = curr_deviation + [angle_diff / 45]  # Append per step, can be max 4

            # Update valid for this step
            new_valid = curr_valid + [is_valid]

            new_coords = curr_coords + [(x_grid[nr, nc], y_grid[nr, nc], z_grid[nr, nc])] # Append new coordinates to list
            new_idx    = curr_idx + [(nr, nc)] # Append new indices to list
            new_steps  = steps + 1
            
            stack.append((new_coords, new_idx, new_steps, new_invalid, new_deviation, new_valid)) # Push new node to paths stack

            if verbose:
                print(
                    f"Pushed node ({nr},{nc}) in paths stack. Current stacks are {len(stack)} and current branch has {new_steps} steps and {new_invalid} consecutive invalid steps ..."
                )

    return paths

# %% === Main function
def main(
        base_dir: str=None, # Base directory for the analysis
        gui_mode: bool=False, # Run in GUI mode
        method: str='gradient', # 'gradient', 'd8-flow', 'slope-compatible'
        flow_sense: str='upstream', # 'upstream', 'downstream'
        starting_source: str='reference_points', # 'reference_points', 'landslides_dataset', 'potential_landslides'
        step_size: int=1, # How many cells to use for each step
        max_steps: int=50, # Maximum number of steps
        min_steps: int=1, # Minimum number of steps
        min_slope_degrees: float=7, # Minimum slope for flow direction
        invalid_steps_tolerance: int=1, # Consecutive number of invalid steps allowed in path
        add_to_existing_paths: bool=True, # Add to existing paths (file in variables folder)
        min_realism_score: float=None, # Optional: Minimum realism score to keep paths (filters out low-score paths early)
        max_paths_per_start: int=None, # Optional: Maximum paths per starting point (keeps top-N by score)
        incremental_save: bool=False, # Save batches to disk incrementally to reduce memory, definitely slower but helpful with memory issues
        cleanup_temp_files: bool=False, # Delete temporary files after completion (only in case of incremental_save)
        early_filter_paths: bool=False # Filter paths also during creation, at an early stage. It can be helpful in case of memory issues, but you loose some paths
    ) -> dict[str, object]:
    """Main function to create landslide paths."""

    # Input validation
    if method not in POSSIBLE_METHODS:
        log_and_error(f"Invalid method: {method}. Must be one of {POSSIBLE_METHODS}.", ValueError, logger)
    if flow_sense not in POSSIBLE_FLOW_SENSES:
        log_and_error(f"Invalid flow_sense: {flow_sense}. Must be one of {POSSIBLE_FLOW_SENSES}.", ValueError, logger)
    if starting_source not in POSSIBLE_STARTING_SOURCES:
        log_and_error(f"Invalid starting_source: {starting_source}. Must be one of {POSSIBLE_STARTING_SOURCES}.", ValueError, logger)
    if step_size <= 0:
        log_and_error("step_size must be positive.", ValueError, logger)
    if not isinstance(max_steps, int) or max_steps < min_steps:
        log_and_error("max_steps must be integer and >= min_steps.", ValueError, logger)
    if not isinstance(min_steps, int) or min_steps <= 0:
        log_and_error("min_steps must be positive and integer.", ValueError, logger)
    if min_slope_degrees <= 0:
        log_and_error("min_slope_degrees must be positive.", ValueError, logger)
    if not isinstance(invalid_steps_tolerance, int) or invalid_steps_tolerance < 0:
        log_and_error("invalid_steps_tolerance must be non-negative and integer.", ValueError, logger)
    if min_realism_score is not None and (min_realism_score < 0 or min_realism_score > 100):
        log_and_error("min_realism_score must be between 0 and 100 if provided.", ValueError, logger)
    if max_paths_per_start is not None and (not isinstance(max_paths_per_start, int) or max_paths_per_start <= 0):
        log_and_error("max_paths_per_start must be positive and integer, if provided.", ValueError, logger)
    
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    if incremental_save:
        TEMP_DIR = os.path.join(env.folders['variables']['path'], 'temp')
        os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure temp dir exists

    dtm_vars = env.load_variable(variable_filename='dtm_vars.pkl')

    abg_df = dtm_vars['abg']
    dtm_df = dtm_vars['dtm']

    if VARIABLE_FILENAME in env.config['variables'].keys() and add_to_existing_paths:
        landslide_paths_vars = env.load_variable(variable_filename=VARIABLE_FILENAME)
        landslide_paths_df = landslide_paths_vars['paths_df']
        landslide_paths_settings = landslide_paths_vars['settings']
    else:
        landslide_paths_vars = {}
        landslide_paths_df = pd.DataFrame()
        landslide_paths_settings = {}
    
    curr_lnd_paths_settings = {
        'method': method,
        'flow_sense': flow_sense,
        'min_steps': min_steps,
        'max_steps': max_steps,
        'step_size': step_size,
        'min_slope_degrees': min_slope_degrees,
        'invalid_steps_tolerance': invalid_steps_tolerance
    }

    if curr_lnd_paths_settings not in landslide_paths_settings.values():
        curr_sett_id = f"lps_{len(landslide_paths_settings) + 1}"
        landslide_paths_settings[curr_sett_id] = curr_lnd_paths_settings
    else:
        curr_sett_id = list(landslide_paths_settings.keys())[list(landslide_paths_settings.values()).index(curr_lnd_paths_settings)]

    if gui_mode:
        log_and_error("GUI mode is not supported in this script yet. Please run the script without GUI mode.", NotImplementedError)
    else:
        if starting_source == 'reference_points':
            print("\n=== Reference points file selection ===")
            source_path = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the reference points csv (Default: {REFERENCE_POINTS_FILENAME}): ",
                src_ext=SUPPORTED_FILE_TYPES['table']
            )
        elif starting_source == 'landslides_dataset':
            log_and_error("landslides_dataset starting_source is not implemented yet.", NotImplementedError)
        elif starting_source == 'potential_landslides':
            log_and_error("potential_landslides starting_source is not implemented yet.", NotImplementedError)
        else:
            log_and_error(f"Unknown starting_source: {starting_source}", ValueError)

    # Get starting points
    logger.info(f"Loading starting points from {source_path}")
    starting_points_df = get_starting_points(env=env, starting_source=starting_source, file_path=source_path)

    logger.info("Converting ABG and reference points to projected...")
    abg_prj_df, starting_points_prj_df, prj_epsg_code = convert_abg_and_ref_points_to_prj( # Both DataFrames are now projected and contain prj_x and prj_y columns
        abg_df=abg_df,
        ref_points_df=starting_points_df
    )

    del abg_df # No more needed, free up memory

    logger.info("Finding closest DTM point for each reference point...")
    ref_points_dtm, ref_points_1d_idx, ref_points_dist_to_grid = get_closest_grid_and_1d_pixel_idx(
        x=starting_points_prj_df['prj_x'], 
        y=starting_points_prj_df['prj_y'], 
        x_grids=abg_prj_df['prj_x'],
        y_grids=abg_prj_df['prj_y'],
        skip_out_of_bbox=True,
        fill_value=np.nan
    )

    if incremental_save and len(starting_points_df) >= 1000:
        log_and_error("Number of starting points exceeds 3 digits. Please disable incremental_save mode or contact the developer.", ValueError, logger)
    
    # Generate paths
    logger.info("Generating paths...")
    gen_paths_dict_list = []  # Keep small; reset per batch
    temp_file_counter = 0
    path_id_counter = len(landslide_paths_df)  # Global counter for unique path_id
    for r, sp_row in starting_points_df.iterrows():
        sp_count_str = f"{r} of {len(starting_points_df)}"
        curr_dtm, curr_1d_idx, curr_dist_to_grid_point = ref_points_dtm[r], ref_points_1d_idx[r], ref_points_dist_to_grid[r]
        if np.isnan(curr_1d_idx):
            logger.warning(f"No DTM point was found for starting point {sp_count_str} ...")
            continue
        else:
            curr_dtm = int(curr_dtm)
            curr_1d_idx = int(curr_1d_idx)
        
        logger.info(f"Processing starting point {sp_count_str} ...")
        sp_id = sp_row['id']
        sp_lon = sp_row['lon']
        sp_lat = sp_row['lat']

        sp_ref_str = f"[id: {sp_id}] (lon: {sp_lon}, lat: {sp_lat})"

        curr_x_grid = abg_prj_df['prj_x'].iloc[curr_dtm]
        curr_y_grid = abg_prj_df['prj_y'].iloc[curr_dtm]
        curr_z_grid = dtm_df['elevation'].iloc[curr_dtm]

        start_idx = get_2d_idx_from_1d_idx(indices=curr_1d_idx, shape=curr_z_grid.shape)
        
        logger.info(f"Generating path(s) for starting point {sp_count_str} {sp_ref_str} ...")
        if method in ['gradient', 'd8-flow']:
            path_prj_coords, path_idx, steps_count, stop_reason, step_deviation, step_validity = generate_steepsets_path(
                start_idx=start_idx, 
                x_grid=curr_x_grid, 
                y_grid=curr_y_grid, 
                z_grid=curr_z_grid, 
                method=method, 
                flow=flow_sense, 
                step_size=step_size, 
                max_steps=max_steps, 
                min_slope=np.tan(np.deg2rad(min_slope_degrees)), # Slope is a percentage (tg(a)), not degrees, not radians
                invalid_steps_tolerance=invalid_steps_tolerance
            )
            compatible_paths = [(path_prj_coords, path_idx, steps_count, stop_reason, step_deviation, step_validity)]
        elif method == 'slope-compatible':
            # NOTE 1: This function returns a list of all compatible paths, but it can be extremely slow. 
            #         This is in case of large max_steps, large invalid_steps_tolerance, or small min_slope_degrees.
            # NOTE 2: If we think in a phisically possible scenario, if max_steps increases, then min_slope_degrees should also increase. 
            #         In other words, if max_steps is 500 and grid is 10x10 meters, it would be unreasonable to have min_slope_degrees 
            #         smaller than 10 degrees (for instance), because this means that landslides can have a very long runout distance 
            #         in a almost flat area, before they stop.
            compatible_paths = generate_slope_compatible_paths(
                start_idx=start_idx, 
                x_grid=curr_x_grid, 
                y_grid=curr_y_grid, 
                z_grid=curr_z_grid, 
                flow=flow_sense, 
                step_size=step_size, 
                max_steps=max_steps, 
                min_slope=np.tan(np.deg2rad(min_slope_degrees)), # min_slope is a percentage (tg(a)), not degrees, not radians
                invalid_steps_tolerance=invalid_steps_tolerance,
                maximum_paths=MAXIMUM_COMPATIBLE_PATHS,
                allow_low_slope_opposite_flow=False,
                min_partial_realism_score=min_realism_score,  # Use min_realism_score as threshold for partial score
                verbose=False
            )

            if len(compatible_paths) >= MAXIMUM_COMPATIBLE_PATHS:
                logger.warning(f"Number of paths is greater than maximum allowed for starting point {sp_count_str} {sp_ref_str}. Paths exploration was stopped and only the first {MAXIMUM_COMPATIBLE_PATHS} paths will be used.")
        
        memory_report(logger)
        logger.info(f"Filtering paths for starting point {sp_count_str} {sp_ref_str} ...")

        # Collect all path_prj_coords for batch conversion
        all_path_prj_coords = []
        path_indices = []  # To track start/end indices for each path
        start_idx = 0
        for path_prj_coords, path_idx, steps_count, stop_reason, step_deviation, step_validity in compatible_paths:
            if steps_count >= min_steps: # Maintain this condition here and below
                all_path_prj_coords.append(path_prj_coords)
                path_indices.append((start_idx, start_idx + len(path_prj_coords)))
                start_idx += len(path_prj_coords)
        
        if len(all_path_prj_coords) == 0:
            logger.warning(f"No compatible paths remaining for starting point {sp_count_str} {sp_ref_str} ...")
            continue
        elif len(all_path_prj_coords) < len(compatible_paths):
            logger.info(f"Remaining paths are {len(all_path_prj_coords)} and {len(compatible_paths) - len(all_path_prj_coords)} paths containing less than {min_steps} steps were filtered out for starting point {sp_count_str} {sp_ref_str} ...")
        else:
            logger.info(f"Remaining paths are {len(all_path_prj_coords)} and no paths were filtered out for starting point {sp_count_str} {sp_ref_str} ...")
        
        # Concatenate all coords for batch conversion
        all_coords = np.concatenate(all_path_prj_coords, axis=0)
        all_lon, all_lat = convert_coords(crs_in=prj_epsg_code, crs_out=4326, in_coords_x=all_coords[:, 0], in_coords_y=all_coords[:, 1])
        
        # Additional info for each path
        logger.info(f"Calculating additional info and realism score for paths of starting point {sp_count_str} {sp_ref_str} ...")
        filtered_paths = []
        path_counter = 0
        for path_prj_coords, path_idx, steps_count, stop_reason, step_deviation, step_validity in compatible_paths:
            if steps_count >= min_steps: # Maintain this condition here and above
                # Pre-calculate diffs once
                diff_xy = np.diff(path_prj_coords[:, 0:2], axis=0)
                diff_xyz = np.diff(path_prj_coords, axis=0)
                diff_z = np.diff(path_prj_coords[:, 2], axis=0)
                
                path_step_plane_length = np.sqrt(np.sum(diff_xy ** 2, axis=1))
                path_step_tridim_length = np.sqrt(np.sum(diff_xyz ** 2, axis=1))
                
                path_plane_progressive_length = np.concatenate([[0], np.cumsum(path_step_plane_length)])
                path_tridim_progressive_length = np.concatenate([[0], np.cumsum(path_step_tridim_length)])
                
                path_step_slope_degrees = np.rad2deg(np.arctan2(diff_z, path_step_plane_length))
                
                # Calculate realism score: higher is more realistic
                valid_steps_ratio = np.mean(step_validity) if len(step_validity) > 0 else 0
                path_length = path_plane_progressive_length[-1]
                if stop_reason in [STOP_REASONS['loop'], STOP_REASONS['edge']]:
                    stop_penalty = 3
                elif stop_reason in [STOP_REASONS['opposite'], STOP_REASONS['invalid'], STOP_REASONS['low_realism']]:
                    stop_penalty = 2
                elif stop_reason in [STOP_REASONS['max_steps'], STOP_REASONS['max_paths']]:
                    stop_penalty = 1
                elif stop_reason in [STOP_REASONS['complete']]:
                    stop_penalty = 0
                else:
                    log_and_error(f"Not implemented stop reason in penalty: {stop_reason}", ValueError)
                
                avg_slope = np.mean(path_step_slope_degrees) if len(path_step_slope_degrees) > 0 else 0
                average_slope_penalty = min(60, max(0, (min_slope_degrees - abs(avg_slope)) * 5)) # Penalize if average slope is below min. Each degree of difference (below min) is 5 points, maximum penalty is 60 points
                avg_deviation = np.mean(step_deviation) if len(step_deviation) > 0 else 0 # it can be at max 4
                realism_score = max(0, (valid_steps_ratio * 90) + min(path_length / 5, 10) - (stop_penalty * 2) - (avg_deviation * 5) - average_slope_penalty) # Maximum score is 100 (because (valid_steps_ratio * 90) + min(path_length / 5, 10) is at max 100), min score is 0 (if first part is below 100, then the penalties can make it below 0)
                
                # Early filtering: check min_realism_score and collect only if passes
                if early_filter_paths and min_realism_score is not None and realism_score < min_realism_score:
                    pass # Skip to next path
                else:
                    # Extract geo coords from batch conversion
                    start, end = path_indices[path_counter]
                    path_lon = all_lon[start:end]
                    path_lat = all_lat[start:end]
                    path_geo_coords = path_prj_coords.copy()
                    path_geo_coords[:, 0] = path_lon
                    path_geo_coords[:, 1] = path_lat
                    
                    filtered_paths.append({
                        'path_id': f"PLP_{path_id_counter}", # Use global counter for unique path_id. PLP = Potential Landslides Path
                        'starting_source': starting_source,
                        'starting_point_id': sp_id,
                        'path_realism_score': realism_score,
                        'steps_count': steps_count,
                        'stop_reason': stop_reason,
                        'path_dtm': curr_dtm,
                        'start_snap_offset_meters': curr_dist_to_grid_point,
                        'path_2D_idx': path_idx,
                        'path_geo_coords': path_geo_coords,
                        'path_2D_progr_len_meters': path_plane_progressive_length,
                        'path_3D_progr_len_meters': path_tridim_progressive_length,
                        'path_step_slope_degrees': path_step_slope_degrees,
                        'path_step_deviation': step_deviation,
                        'path_step_validity': step_validity
                    })

                    path_id_counter += 1  # Increment counter after adding each path to ensure unique IDs

                path_counter += 1 # Must be out of if/else block above, because it's very important in both cases (wheter it passes or not the early filter) because it keeps track of the path index
        
        # Limit paths per starting point if specified
        if early_filter_paths and max_paths_per_start is not None and len(filtered_paths) > max_paths_per_start:
            filtered_paths = sorted(filtered_paths, key=lambda x: x['path_realism_score'], reverse=True)[:max_paths_per_start]
            logger.info(f"Limited to top {max_paths_per_start} paths by realism score for starting point {sp_count_str} {sp_ref_str} ...")

        # Batch build and save
        if filtered_paths:
            batch_df = pd.DataFrame(filtered_paths)
            batch_df['setting_id'] = curr_sett_id  # Add setting_id here
            gen_paths_dict_list.extend(filtered_paths)  # For final concat if not incremental

            logger.info(f"Saving batch {temp_file_counter}, with {len(filtered_paths)} paths for starting point {sp_count_str} {sp_ref_str} ...")
            if incremental_save:
                if JOBLIB_AVAILABLE and max_steps > 5000: # For large arrays, it could be better to use joblib
                    temp_file = os.path.join(TEMP_DIR, f"{VARIABLE_FILENAME}_temp_{temp_file_counter:03d}.jbl")
                    joblib.dump(batch_df, temp_file, compress=3) # Use joblib with compression
                else:
                    temp_file = os.path.join(TEMP_DIR, f"{VARIABLE_FILENAME}_temp_{temp_file_counter:03d}.pkl")
                    batch_df.to_pickle(temp_file, compression='gzip') # Use Pandas to_pickle with gzip compression
                temp_file_counter += 1
                logger.info(f"Saved batch to temp file: {temp_file} ...")
                memory_report(logger)

                # Clear memory-intensive variables after saving batch
                del filtered_paths, batch_df, compatible_paths, all_path_prj_coords, path_indices, all_coords, all_lon, all_lat
                gen_paths_dict_list.clear()  # Reset since data is saved to disk
            else:
                memory_report(logger)
                pass # If not incremental save, keep gen_paths_dict_list for later concat. NOTE: High memory usage

    # Final concatenation
    if incremental_save:
        logger.info(f"Starting final concatenation of temp files...")
        memory_report(logger)

        temp_files_jbl = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.startswith(f"{VARIABLE_FILENAME}_temp_") and f.endswith('.jbl')]
        temp_files_pkl = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.startswith(f"{VARIABLE_FILENAME}_temp_") and f.endswith('.pkl')]
        temp_files = temp_files_jbl + temp_files_pkl
        if temp_files:
            curr_paths_df = pd.DataFrame()
            for f in temp_files:
                logger.info(f"Reading temp file: {f} ...")
                if f.endswith('.jbl'):
                    curr_loaded_df = joblib.load(f) # Load with joblib
                elif f.endswith('.pkl'):
                    curr_loaded_df = pd.read_pickle(f, compression='gzip')  # Load Pickle with gzip
                else:
                    log_and_error(f"Unknown file type for temp file: {f}", ValueError)
                
                # Filter paths
                if min_realism_score is not None:
                    min_rl_scr_mask = curr_loaded_df['path_realism_score'] >= min_realism_score
                    curr_loaded_df = curr_loaded_df[min_rl_scr_mask]
                    logger.info(f"Filtered out {len(curr_loaded_df) - min_rl_scr_mask.sum()} paths with realism score below {min_realism_score} in temp file {f} ...")
                if max_paths_per_start is not None:
                    curr_loaded_df.sort_values(by='path_realism_score', ascending=False, inplace=True)
                    curr_loaded_df.reset_index(drop=True, inplace=True)
                    logger.info(f"Filtered out {len(curr_loaded_df) - max_paths_per_start} paths with lower realism score in temp file {f} ...")
                    curr_loaded_df = curr_loaded_df.iloc[:max_paths_per_start]
                
                curr_paths_df = pd.concat([curr_paths_df, curr_loaded_df], ignore_index=True)
                memory_report(logger)
            
            if cleanup_temp_files: # Clean up temp files
                for f in temp_files:
                    os.remove(f)
                os.rmdir(TEMP_DIR)
        else:
            log_and_error(f"No temp files found, please check if they exist and that names are correct (e.g. [{VARIABLE_FILENAME}_temp_*.jbl or *.pkl])", ValueError)
    else:
        curr_paths_df = pd.DataFrame(gen_paths_dict_list)
    
    curr_paths_df['setting_id'] = curr_sett_id

    logger.info(f"Generated {len(curr_paths_df)} potential landslide paths (PLPs) for all the starting points...")

    landslide_paths_df = pd.concat([landslide_paths_df, curr_paths_df], ignore_index=True)

    # Sort by realism score descending (higher realism first)
    landslide_paths_df = landslide_paths_df.sort_values(by='path_realism_score', ascending=False).reset_index(drop=True)

    landslide_paths_vars['paths_df'] = landslide_paths_df
    landslide_paths_vars['settings'] = landslide_paths_settings

    env.save_variable(variable_to_save=landslide_paths_vars, variable_filename=VARIABLE_FILENAME, compression='gzip')

    return landslide_paths_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate landslide paths (steepests or slope-compatible).")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    parser.add_argument("--method", type=str, default='gradient', help="Path generation method (gradient, d8-flow, slope-compatible)")
    parser.add_argument("--flow_sense", type=str, default='upstream', help="Flow sense (upstream, downstream)")
    parser.add_argument("--step_size", type=int, default=1, help="How many cells to move at each step")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of steps for the generated path")
    parser.add_argument("--min_steps", type=int, default=2, help="Minimum number of steps for the generated path")
    parser.add_argument("--min_slope_degrees", type=float, default=5, help="Minimum slope in degrees for the generated path")
    parser.add_argument("--invalid_steps_tolerance", type=int, default=1, help="Tolerance for invalid consecutive steps in the generated path")
    parser.add_argument("--add_to_existing_paths", action="store_true", help="Add generated paths to existing landslide paths (in variables folder)")
    parser.add_argument("--min_realism_score", type=float, help="Optional: Minimum realism score to keep paths")
    parser.add_argument("--max_paths_per_start", type=int, help="Optional: Maximum paths per starting point")
    parser.add_argument("--incremental_save", action="store_true", default=True, help="Save batches to disk incrementally")
    parser.add_argument("--cleanup_temp_files", action="store_true", default=True, help="Clean up temp files after concatenation")
    parser.add_argument("--early_filter_paths", action="store_true", default=True, help="Early filter paths based on score or maximum paths per starting point")
    
    args = parser.parse_args()
    
    landslide_paths_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        method=args.method,
        flow_sense=args.flow_sense,
        step_size=args.step_size,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        min_slope_degrees=args.min_slope_degrees,
        invalid_steps_tolerance=args.invalid_steps_tolerance,
        add_to_existing_paths=args.add_to_existing_paths,
        min_realism_score=args.min_realism_score,
        max_paths_per_start=args.max_paths_per_start,
        incremental_save=args.incremental_save,
        cleanup_temp_files=args.cleanup_temp_files,
        early_filter_paths=args.early_filter_paths
    )