# %% === Import necessary modules
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist  # For efficient distance calculations in slope checks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    SUPPORTED_FILE_TYPES,
    REFERENCE_POINTS_FILENAME
)

# # Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

from psliptools.utilities import (
    select_file_prompt,
    read_generic_csv
)

# from psliptools.geometries import (
# )

# from psliptools.scattered import (
# )

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
from main_modules.m05a_reference_points_info import convert_abg_and_ref_points_to_prj, get_closest_dtm_point
logger = setup_logger(__name__)
logger.info("=== Landslide paths creation ===")

# %% === Helper functions
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

def create_grid_mapping(
        x_grid: np.ndarray, 
        y_grid: np.ndarray
    ) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Create a mapping for quick index lookup."""
    unique_x = x_grid[0, :]  # Longitudes
    unique_y = y_grid[:, 0]  # Latitudes

    # Mappings for quick index lookup
    x_to_idx = {x: i for i, x in enumerate(unique_x)}
    y_to_idx = {y: i for i, y in enumerate(unique_y)}

    return x_to_idx, y_to_idx, unique_x, unique_y

def get_neighbors(row, col, grid_shape):
    """Get 8 possible neighbors (D8 directions)."""
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
            neighbors.append((nr, nc))
    return neighbors

def calculate_slope(elev1, elev2, step_size):
    """Calculate slope in degrees between two elevations."""
    if step_size == 0:
        return 0
    return np.degrees(np.arctan((elev2 - elev1) / step_size))

def generate_path(start_idx, start_coords, grid, x_to_idx, y_to_idx, unique_x, unique_y, method, flow_sense, step_size, max_runout, min_runout, min_slope_degrees, invalid_steps_tolerance):
    """Generate a single path from start point."""
    path_coords = [start_coords]
    path_idx = [start_idx]
    steps_count = 0
    invalid_steps = 0
    stop_reason = None
    current_row, current_col = start_idx
    
    while steps_count < max_runout:
        if method == 'gradient':
            # Steepest ascent/descent
            neighbors = get_neighbors(current_row, current_col, grid.shape)
            if not neighbors:
                stop_reason = 'edge_reached'
                break
            elevations = [grid[nr, nc] for nr, nc in neighbors]
            if flow_sense == 'downstream':
                best_idx = np.argmin(elevations)  # Lowest elevation
            else:  # upstream
                best_idx = np.argmax(elevations)  # Highest elevation
            next_row, next_col = neighbors[best_idx]
            slope = calculate_slope(grid[current_row, current_col], grid[next_row, next_col], step_size)
            if slope < min_slope_degrees:
                invalid_steps += 1
            else:
                invalid_steps = 0
        elif method == 'd8-flow':
            # D8: Choose direction with steepest drop/rise
            neighbors = get_neighbors(current_row, current_col, grid.shape)
            if not neighbors:
                stop_reason = 'edge_reached'
                break
            slopes = [calculate_slope(grid[current_row, current_col], grid[nr, nc], step_size) for nr, nc in neighbors]
            if flow_sense == 'downstream':
                best_idx = np.argmin(slopes)  # Steepest descent
            else:
                best_idx = np.argmax(slopes)  # Steepest ascent
            next_row, next_col = neighbors[best_idx]
            slope = slopes[best_idx]
            if slope < min_slope_degrees:
                invalid_steps += 1
            else:
                invalid_steps = 0
        elif method == 'slope-compatible':
            # This will be handled separately for branching
            break  # Placeholder; full implementation below
        
        if invalid_steps > invalid_steps_tolerance:
            stop_reason = 'slope_tolerance_exceeded'
            break
        
        # Move to next point
        current_row, current_col = next_row, next_col
        path_coords.append((unique_x[current_col], unique_y[current_row], grid[current_row, current_col]))
        path_idx.append((current_row, current_col))
        steps_count += 1
    
    if steps_count >= max_runout:
        stop_reason = 'max_runout'
    return path_coords, path_idx, steps_count, stop_reason or 'completed'

def generate_slope_compatible_paths(start_idx, start_coords, grid, x_to_idx, y_to_idx, unique_x, unique_y, flow_sense, step_size, max_runout, min_slope_degrees, invalid_steps_tolerance):
    """Generate multiple slope-compatible paths (recursive branching)."""
    paths = []
    def explore(current_row, current_col, current_path_coords, current_path_idx, steps_count, invalid_steps):
        if steps_count >= max_runout or invalid_steps > invalid_steps_tolerance:
            stop_reason = 'max_runout' if steps_count >= max_runout else 'slope_tolerance_exceeded'
            paths.append((current_path_coords[:], current_path_idx[:], steps_count, stop_reason))
            return
        neighbors = get_neighbors(current_row, current_col, grid.shape)
        valid_neighbors = []
        for nr, nc in neighbors:
            slope = calculate_slope(grid[current_row, current_col], grid[nr, nc], step_size)
            if (flow_sense == 'downstream' and slope >= min_slope_degrees) or (flow_sense == 'upstream' and slope <= -min_slope_degrees):
                valid_neighbors.append((nr, nc))
        if not valid_neighbors:
            paths.append((current_path_coords[:], current_path_idx[:], steps_count, 'no_valid_neighbors'))
            return
        for nr, nc in valid_neighbors:
            new_path_coords = current_path_coords + [(unique_x[nc], unique_y[nr], grid[nr, nc])]
            new_path_idx = current_path_idx + [(nr, nc)]
            explore(nr, nc, new_path_coords, new_path_idx, steps_count + 1, 0)  # Reset invalid_steps on valid move
    explore(start_idx[0], start_idx[1], [start_coords], [start_idx], 0, 0)
    return paths

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        method: str='gradient', # 'gradient', 'd8-flow', 'slope-compatible'
        flow_sense: str='upstream', # 'upstream', 'downstream'
        starting_source: str='reference_points', # 'reference_points', 'landslides_dataset', 'potential_landslides'
        step_size: float=1.0,
        max_steps: int=500, # Maximum number of steps
        min_runout: int=0, # Minimum number of steps
        min_slope_degrees: float=5, # Minimum slope for flow direction
        invalid_steps_tolerance: int=2 # Maximum number of invalid steps
    ) -> dict[str, object]:
    """Main function to create landslide paths."""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    dtm_vars = env.load_variable(variable_filename='dtm_vars.pkl')

    abg_df = dtm_vars['abg']
    dtm_df = dtm_vars['dtm']

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
    
    # Generate paths
    logger.info("Generating paths...")
    paths_data = []
    path_id = 0
    for sp_row_idx, sp_row in starting_points_df.iterrows():
        sp_id = sp_row['id']
        sp_x = sp_row['lon']
        sp_y = sp_row['lat']

        logger.info(f"Getting closest DTM point for starting point {sp_id} ({sp_x}, {sp_y})")
        curr_dtm, curr_1d_idx, curr_dist_to_grid_point = get_closest_dtm_point(
            x=sp_x,
            y=sp_y, 
            base_grid_df=abg_df, 
            base_grid_x_col='longitude', 
            base_grid_y_col='latitude'
        )

        # TODO: Continue revision of code from here (you should actually use all the possible values in the grid, not just first row/column!)
        x_to_idx, y_to_idx, unique_x, unique_y = create_grid_mapping(
            x_grid=abg_df['longitude'].iloc[curr_dtm], 
            y_grid=abg_df['latitude'].iloc[curr_dtm]
        )

        start_row = y_to_idx[sp_y]
        start_col = x_to_idx[sp_x]
        start_idx = (start_row, start_col)
        
        logger.info(f"Generating path(s) for starting point {sp_id} ({sp_x}, {sp_y})")
        if method in ['gradient', 'd8-flow']:
            path_coords, path_idx, steps_count, stop_reason = generate_path(
                start_idx, sp_row, grid, x_to_idx, y_to_idx, unique_x, unique_y,
                method, flow_sense, step_size, max_steps, min_runout, min_slope_degrees, invalid_steps_tolerance
            )
            paths_data.append({
                'path_id': path_id,
                'starting_point_id': sp_id,
                'starting_coords': sp_row,
                'starting_idx': start_idx,
                'starting_source': starting_source,
                'method': method,
                'flow_sense': flow_sense,
                'steps_count': steps_count,
                'stop_reason': stop_reason,
                'path_coords': path_coords,
                'path_idx': path_idx
            })
            path_id += 1
        elif method == 'slope-compatible':
            compatible_paths = generate_slope_compatible_paths(
                start_idx, sp_row, grid, x_to_idx, y_to_idx, unique_x, unique_y,
                flow_sense, step_size, max_steps, min_slope_degrees, invalid_steps_tolerance
            )
            for path_coords, path_idx, steps_count, stop_reason in compatible_paths:
                paths_data.append({
                    'path_id': path_id,
                    'starting_point_id': i,
                    'starting_coords': sp_row,
                    'starting_idx': start_idx,
                    'starting_source': starting_source,
                    'method': method,
                    'flow_sense': flow_sense,
                    'steps_count': steps_count,
                    'stop_reason': stop_reason,
                    'path_coords': path_coords,
                    'path_idx': path_idx
                })
                path_id += 1
    
    paths_df = pd.DataFrame(paths_data)
    return {'paths_df': paths_df}

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="**********Summary of the module**********")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    
    args = parser.parse_args()
    
    time_sensitive_vars = main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode
    )