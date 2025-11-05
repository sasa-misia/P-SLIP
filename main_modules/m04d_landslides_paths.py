# %% === Import necessary modules
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from functools import lru_cache
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    convert_coords
)

from psliptools.utilities import (
    select_file_prompt,
    read_generic_csv
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
from main_modules.m05a_reference_points_info import convert_abg_and_ref_points_to_prj, get_closest_dtm_point
logger = setup_logger(__name__)
logger.info("=== Landslide paths creation ===")

# %% === Helper functions and global variables
MAXIMUM_COMPATIBLE_PATHS = 2000000
VARIABLE_FILENAME = "landslide_paths_vars.pkl"

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
        raise NotImplementedError("landslides_dataset starting_source is not implemented yet.")
    elif starting_source == 'potential_landslides':
        raise NotImplementedError("potential_landslides starting_source is not implemented yet.")
    else:
        raise ValueError(f"Unknown starting_source: {starting_source}")
    
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
                raise ValueError(f"Unknown flow type: {flow}")
            
            # Update position with learning rate (step_size)
            curr_x = curr_x + 10 * dx * step_size * direction_x # 10 is just a learning rate to speed up the process
            curr_y = curr_y + 10 * dy * step_size * direction_y # 10 is just a learning rate to speed up the process
            
            # Check if new position is within grid bounds
            x_min, x_max = np.min(x_grid), np.max(x_grid)
            y_min, y_max = np.min(y_grid), np.max(y_grid)
            if not (x_min <= curr_x <= x_max and y_min <= curr_y <= y_max):
                stop_reason = 'edge_reached'
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
            step_deviation_list.append(angle_diff / np.deg2rad(45))

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
                stop_reason = 'edge_reached'
                break

            if flow == 'downstream':
                best_idx = np.argmin(slopes)  # Steepest descent
            elif flow == 'upstream':
                best_idx = np.argmax(slopes)  # Steepest ascent
            else:
                raise ValueError(f"Unknown flow type: {flow}")
            
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
                    stop_reason = 'slope_greater_than_tolerance_in_opposite_direction'
                    break
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # --- Check stopping criteria
        if (next_row, next_col) in path_2D_idx_list: # Check for loop (bouncing back and forth)
            stop_reason = 'loop_detected'
            break
        
        if invalid_steps > invalid_steps_tolerance: # Check invalid steps tolerance, just > because this invalid point has not been added yet
            stop_reason = 'invalid_steps_tolerance_exceeded'
            break
        
        # --- Move to next point
        curr_row, curr_col = next_row, next_col
        path_coord_list.append((x_grid[curr_row, curr_col], y_grid[curr_row, curr_col], z_grid[curr_row, curr_col]))
        path_2D_idx_list.append((curr_row, curr_col))
        steps_count += 1
    
    # --- Check stopping criteria
    if steps_count >= max_steps: # Check max runout
        stop_reason = 'max_runout'

    if stop_reason is None: # Check if path completed
        stop_reason = 'completed'

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
        maximum_paths: int = 50000,
        allow_low_slope_opposite_flow: bool = False,
        verbose: bool = False
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
        verbose (bool, optional): Whether to print progress updates. Defaults to False.

    Returns:
        list[tuple[np.ndarray, np.ndarray, int, str, np.ndarray, np.ndarray]]: A list of tuples containing the path coordinates, path 2D index, number of steps, stop reason, deviation array, valid steps array.
    """

    @lru_cache(maxsize=None)
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
            warnings.warn(f"Maximum paths limit exceeded! Completed paths: {len(paths)}, remaining paths to explore: {len(stack)}. Stopping exploration...", stacklevel=2)
            for coord, idx, steps, _, dev, valid in stack:
                paths.append((np.array(coord), np.array(idx), steps, 'partial_because_max_paths_exceeded', np.array(dev), np.array(valid)))
            if verbose:
                print(
                    f"Exploration was stopped because the maximum number of paths was exceeded. Completed paths: {len(paths) - len(stack)}, remaining paths to explore: {len(stack)}."
                )
            break

        # --- early termination: max runout ----------
        if steps >= max_steps:
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, 'max_runout', np.array(curr_deviation), np.array(curr_valid)))
            if verbose:
                print(f"Path n. {len(paths)} has reached max runout; stopped at step: {steps} ...")
            continue
        # --------------------------------------------

        slopes_df = _cached_neighbors(curr_idx[-1][0], curr_idx[-1][1], step_size)
        neighbors = slopes_df[['row_end', 'col_end']].values
        slopes = slopes_df['slope'].values

        # --- early termination: edge reached --------
        if np.isnan(slopes).any():
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, 'edge_reached', np.array(curr_deviation), np.array(curr_valid)))
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
            raise ValueError(f"Invalid flow sense: {flow}")

        compatible_mask = valid_mask | low_mask # NOTE: not compatible_mask represents neighbors with the opposite flow with sloped terrain (greater than min_slope and in opposite flow direction)

        # --- early termination: all neighbors with opposite flow ----------
        if not compatible_mask.any():
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, 'all_neighbors_with_opposite_flow', np.array(curr_deviation), np.array(curr_valid)))
            if verbose:
                print(f"Path n. {len(paths)} has reached all neighbors with opposite flow; stopped at step: {steps} ...")
            continue
        # ------------------------------------------------------------------

        # --- early termination: invalid steps tolerance exceeded with all compatible neighbors ----------
        if not valid_mask.any() and low_mask.any() and (invalid == invalid_steps_tolerance): # All compatible neighbors would be other invalid points, thus exceeding tolerance
            paths.append((np.array(curr_coords), np.array(curr_idx), steps, 'invalid_steps_tolerance_exceeded', np.array(curr_deviation), np.array(curr_valid)))
            if verbose:
                print(f"path n. {len(paths)} has reached all compatible neighbors with consecutive invalid steps tolerance; stopped at step: {steps} ...")
            continue
        # ------------------------------------------------------------------------------------------------


        # Directions: 0=E, 45=SE, 90=S, 135=SW, 180=W, 225=NW, 270=N, 315=NE
        if (slopes_df.index == ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']).all():
            directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        else:
            raise ValueError(f"Invalid slope directions: {slopes_df.index}")

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
            new_deviation = curr_deviation + [angle_diff / 45]  # Append per step

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
        base_dir: str=None,
        gui_mode: bool=False,
        method: str='gradient', # 'gradient', 'd8-flow', 'slope-compatible'
        flow_sense: str='upstream', # 'upstream', 'downstream'
        starting_source: str='reference_points', # 'reference_points', 'landslides_dataset', 'potential_landslides'
        step_size: int=1, # How many cells to use for each step
        max_steps: int=50, # Maximum number of steps
        min_steps: int=1, # Minimum number of steps
        min_slope_degrees: float=7, # Minimum slope for flow direction
        invalid_steps_tolerance: int=1, # Consecutive number of invalid steps allowed in path
        add_to_existing_paths: bool=True # Add to existing paths (file in variables folder)
    ) -> dict[str, object]:
    """Main function to create landslide paths."""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    dtm_vars = env.load_variable(variable_filename='dtm_vars.pkl')

    abg_df = dtm_vars['abg']
    dtm_df = dtm_vars['dtm']

    if VARIABLE_FILENAME in env.config['variables'].keys() and add_to_existing_paths:
        lands_vars = env.load_variable(variable_filename=VARIABLE_FILENAME)
        landslide_paths_df = lands_vars['paths_df']
        landslide_paths_settings = lands_vars['settings']
    else:
        landslide_paths_df = pd.DataFrame()
        landslide_paths_settings = {}

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        if starting_source == 'reference_points':
            print("\n=== Reference points file selection ===")
            source_path = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the reference points csv (Default: {REFERENCE_POINTS_FILENAME}): ",
                src_ext=SUPPORTED_FILE_TYPES['table']
            )
        elif starting_source == 'landslides_dataset':
            raise NotImplementedError("landslides_dataset starting_source is not implemented yet.")
        elif starting_source == 'potential_landslides':
            raise NotImplementedError("potential_landslides starting_source is not implemented yet.")
        else:
            raise ValueError(f"Unknown starting_source: {starting_source}")

    # Get starting points
    logger.info(f"Loading starting points from {source_path}")
    starting_points_df = get_starting_points(env=env, starting_source=starting_source, file_path=source_path)

    logger.info("Converting ABG and reference points to projected...")
    abg_prj_df, starting_points_prj_df, prj_epsg_code = convert_abg_and_ref_points_to_prj( # Both DataFrames are now projected and contain prj_x and prj_y columns
        abg_df=abg_df,
        ref_points_df=starting_points_df
    )
    
    # Generate paths
    logger.info("Generating paths...")
    gen_paths_dict_list = []
    for sp_row_idx, sp_row in starting_points_df.iterrows():
        logger.info(f"Processing starting point {sp_row_idx} of {len(starting_points_df)}")
        sp_id = sp_row['id']
        sp_lon = sp_row['lon']
        sp_lat = sp_row['lat']
        sp_x = starting_points_prj_df['prj_x'][sp_row_idx]
        sp_y = starting_points_prj_df['prj_y'][sp_row_idx]

        logger.info(f"Getting closest DTM point for starting point {sp_id} ({sp_lon}, {sp_lat})")
        curr_dtm, curr_1d_idx, curr_dist_to_grid_point = get_closest_dtm_point(
            x=sp_x,
            y=sp_y, 
            base_grid_df=abg_prj_df, 
            base_grid_x_col='prj_x', 
            base_grid_y_col='prj_y'
        )

        if np.isnan(curr_1d_idx):
            logger.warning(f"No DTM point found for starting point {sp_id} ({sp_lon}, {sp_lat})")
            continue

        curr_x_grid = abg_prj_df['prj_x'].iloc[curr_dtm]
        curr_y_grid = abg_prj_df['prj_y'].iloc[curr_dtm]
        curr_z_grid = dtm_df['elevation'].iloc[curr_dtm]

        start_idx = get_2d_idx_from_1d_idx(indices=curr_1d_idx, shape=curr_z_grid.shape)
        
        logger.info(f"Generating path(s) for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y})")
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
                verbose=False
            )

            if len(compatible_paths) >= MAXIMUM_COMPATIBLE_PATHS:
                logger.warning(f"Number of paths for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y}) is greater than maximum allowed. Paths exploration was stopped and only the first {MAXIMUM_COMPATIBLE_PATHS} paths will be used.")
        
        logger.info(f"Filtering paths and extracting more information for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y})")

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
            logger.warning(f"No compatible paths remaining for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y})")
            continue
        elif len(all_path_prj_coords) < len(compatible_paths):
            logger.info(f"Remaining paths are {len(all_path_prj_coords)} and {len(compatible_paths) - len(all_path_prj_coords)} paths containing less than {min_steps} steps were filtered out for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y})")
        else:
            logger.info(f"Remaining paths are {len(all_path_prj_coords)} for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y}). No paths were filtered out.")
        
        # Concatenate all coords for batch conversion
        all_coords = np.concatenate(all_path_prj_coords, axis=0)
        all_lon, all_lat = convert_coords(crs_in=prj_epsg_code, crs_out=4326, in_coords_x=all_coords[:, 0], in_coords_y=all_coords[:, 1])
        
        # Reset for processing
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
                penalty = 5 if stop_reason in ['loop_detected', 'edge_reached', 'slope_greater_than_tolerance_in_opposite_direction'] else 0
                avg_slope = np.mean(path_step_slope_degrees) if len(path_step_slope_degrees) > 0 else 0
                average_slope_penalty = max(0, (min_slope_degrees - abs(avg_slope)) * 5)  # Penalize if average slope is below min
                avg_deviation = np.mean(step_deviation) if len(step_deviation) > 0 else 0
                realism_score = (valid_steps_ratio * 100) + min(path_length / 100, 10) - penalty - (avg_deviation * 2) - average_slope_penalty
                
                # Extract geo coords from batch conversion
                start, end = path_indices[path_counter]
                path_lon = all_lon[start:end]
                path_lat = all_lat[start:end]
                path_geo_coords = path_prj_coords.copy()
                path_geo_coords[:, 0] = path_lon
                path_geo_coords[:, 1] = path_lat
                
                gen_paths_dict_list.append({
                    'path_id': f"PLP_{len(landslide_paths_df) + len(gen_paths_dict_list)}", # Potential Landslide Path PLP
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

                path_counter += 1
    
    curr_paths_df = pd.DataFrame(gen_paths_dict_list)

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
    
    curr_paths_df['setting_id'] = curr_sett_id

    logger.info(f"Generated {len(curr_paths_df)} potential landslide paths (PLPs)")

    landslide_paths_df = pd.concat([landslide_paths_df, curr_paths_df], ignore_index=True)

    # Sort by realism score descending (higher realism first)
    landslide_paths_df = landslide_paths_df.sort_values(by='path_realism_score', ascending=False).reset_index(drop=True)

    landslide_paths_vars = {'paths_df': landslide_paths_df, 'settings': landslide_paths_settings}

    env.save_variable(variable_to_save=landslide_paths_vars, variable_filename=VARIABLE_FILENAME, compression='gzip')

    return landslide_paths_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="**********Summary of the module**********")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    parser.add_argument("--method", type=str, default='gradient', help="Path generation method (gradient, d8-flow, slope-compatible)")
    parser.add_argument("--flow_sense", type=str, default='upstream', help="Flow sense (upstream, downstream)")
    parser.add_argument("--step_size", type=int, default=1, help="How many cells to move at each step")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of steps for the generated path")
    parser.add_argument("--min_steps", type=int, default=2, help="Minimum number of steps for the generated path")
    parser.add_argument("--min_slope_degrees", type=float, default=5, help="Minimum slope in degrees for the generated path")
    parser.add_argument("--invalid_steps_tolerance", type=int, default=1, help="Tolerance for invalid consecutive steps in the generated path")
    parser.add_argument("--add_to_existing_paths", action="store_true", help="Add generated paths to existing landslide paths")
    
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
        add_to_existing_paths=args.add_to_existing_paths
    )