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

# Importing necessary modules from psliptools
from psliptools.rasters import (
    pick_point_from_1d_idx,
    get_2d_idx_from_1d_idx,
    get_d8_neighbors_row_col,
    get_d8_neighbors_slope
)

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

def calculate_slope(elev1, elev2, step_size):
    """Calculate slope in degrees between two elevations."""
    if step_size == 0:
        return 0
    return np.degrees(np.arctan((elev2 - elev1) / step_size))

def calculate_gradient(curr_row, curr_col, z_grid, x_grid, y_grid):
    """Calculate gradient vector (dx, dy) at current position using finite differences."""
    rows, cols = z_grid.shape
    dx = dy = 1.0  # Default step, but use actual distances if available
    if curr_col > 0 and curr_col < cols - 1:
        z_left = z_grid[curr_row, curr_col - 1]
        z_right = z_grid[curr_row, curr_col + 1]
        x_left = x_grid[curr_row, curr_col - 1]
        x_right = x_grid[curr_row, curr_col + 1]
        grad_x = (z_right - z_left) / (x_right - x_left)
    else:
        grad_x = 0.0  # Edge, no gradient
    if curr_row > 0 and curr_row < rows - 1:
        z_up = z_grid[curr_row - 1, curr_col]
        z_down = z_grid[curr_row + 1, curr_col]
        y_up = y_grid[curr_row - 1, curr_col]
        y_down = y_grid[curr_row + 1, curr_col]
        grad_y = (z_down - z_up) / (y_down - y_up)
    else:
        grad_y = 0.0
    return grad_x, grad_y

def find_nearest_grid_idx(x, y, x_grid, y_grid): # TODO: Compare with your implementation from psliptools (get_closest_pixel_idx)
    """Find the nearest (row, col) index to given (x, y) coordinates."""
    coords = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    target = np.array([[x, y]])
    distances = cdist(target, coords)
    min_idx = np.argmin(distances)
    return np.unravel_index(min_idx, x_grid.shape)

def generate_path(
        start_idx: tuple[int, int],
        start_coords: tuple[float, float, float], # x, y, z 
        x_grid: np.ndarray, 
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        method: str, 
        flow: str, 
        step_size: float, 
        max_steps: int, 
        min_slope: float, 
        invalid_steps_tolerance: int
    ) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]], int, str]:
    """Generate a single path from start point."""
    path_coords = [start_coords]
    path_idx = [start_idx]
    steps_count = 0
    invalid_steps = 0
    stop_reason = None
    curr_row, curr_col = start_idx
    
    while steps_count < max_steps:
        if method == 'gradient': # TODO: Check this method
            # Compute gradient at current position
            grad_x, grad_y = calculate_gradient(curr_row, curr_col, z_grid, x_grid, y_grid)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Determine direction: negative gradient for descent, positive for ascent
            if flow == 'downstream':
                direction_x = -grad_x
                direction_y = -grad_y
            else:  # upstream
                direction_x = grad_x
                direction_y = grad_y
            
            # Update position with learning rate (step_size)
            new_x = x_grid[curr_row, curr_col] + step_size * direction_x
            new_y = y_grid[curr_row, curr_col] + step_size * direction_y
            
            # Check if new position is within grid bounds
            x_min, x_max = np.min(x_grid), np.max(x_grid)
            y_min, y_max = np.min(y_grid), np.max(y_grid)
            if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                stop_reason = 'edge_reached'
                break
            
            # Find nearest grid index to new position
            next_row, next_col = find_nearest_grid_idx(new_x, new_y, x_grid, y_grid)
            
            # Check for minimal movement (avoid stalling on flat areas)
            if (next_row, next_col) == (curr_row, curr_col):
                invalid_steps += 1
            elif grad_magnitude < min_slope:  # Use gradient magnitude as slope proxy
                invalid_steps += 1
            else:
                invalid_steps = 0
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
            slopes = slopes_df['slope'].values

            if np.isnan(slopes).any():
                stop_reason = 'edge_reached'
                break

            if flow == 'downstream':
                best_idx = np.argmin(slopes)  # Steepest descent
            else:
                best_idx = np.argmax(slopes)  # Steepest ascent
            
            next_row, next_col = neighbors[best_idx]
            slope = slopes[best_idx]

            if slope < min_slope:
                invalid_steps += 1
            else:
                invalid_steps = 0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Check for loop (bouncing back and forth)
        if (next_row, next_col) in path_idx:
            stop_reason = 'loop_detected'
            break
        
        # Check slope tolerance
        if invalid_steps > invalid_steps_tolerance:
            stop_reason = 'slope_tolerance_exceeded'
            break
        
        # Move to next point
        curr_row, curr_col = next_row, next_col
        path_coords.append((x_grid[curr_row, curr_col], y_grid[curr_row, curr_col], z_grid[curr_row, curr_col]))
        path_idx.append((curr_row, curr_col))
        steps_count += 1
    
    # Check max runout
    if steps_count >= max_steps:
        stop_reason = 'max_runout'
    
    return np.array(path_coords), np.array(path_idx), steps_count, stop_reason or 'completed'

def generate_slope_compatible_paths(start_idx, start_coords, grid, lon_grid, lat_grid, flow_sense, step_size, max_steps, min_slope_degrees, invalid_steps_tolerance, min_steps):
    """Generate multiple slope-compatible paths (recursive branching)."""
    paths = []
    def explore(current_row, current_col, current_path_coords, current_path_idx, steps_count, invalid_steps):
        if steps_count >= max_steps or invalid_steps > invalid_steps_tolerance:
            stop_reason = 'max_runout' if steps_count >= max_steps else 'slope_tolerance_exceeded'
            if steps_count >= min_steps:
                paths.append((current_path_coords[:], current_path_idx[:], steps_count, stop_reason))
            return
        neighbors = get_d8_neighbors_row_col(current_row, current_col, grid.shape)
        valid_neighbors = []
        for nr, nc in neighbors:
            slope = calculate_slope(grid[current_row, current_col], grid[nr, nc], step_size)
            if (flow_sense == 'downstream' and slope >= min_slope_degrees) or (flow_sense == 'upstream' and slope <= -min_slope_degrees):
                valid_neighbors.append((nr, nc))
        if not valid_neighbors:
            if steps_count >= min_steps:
                paths.append((current_path_coords[:], current_path_idx[:], steps_count, 'no_valid_neighbors'))
            return
        for nr, nc in valid_neighbors:
            # Check for loop (bouncing back and forth)
            if (nr, nc) in current_path_idx:
                # Skip this neighbor to avoid loop
                continue
            new_path_coords = current_path_coords + [(lon_grid[nr, nc], lat_grid[nr, nc], grid[nr, nc])]
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
        min_steps: int=0, # Minimum number of steps
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

        curr_x_grid = abg_df['longitude'].iloc[curr_dtm]
        curr_y_grid = abg_df['latitude'].iloc[curr_dtm]
        curr_z_grid = dtm_df['elevation'].iloc[curr_dtm]
        sp_x_from_grid = pick_point_from_1d_idx(raster=curr_x_grid, idx_1d=curr_1d_idx)
        sp_y_from_grid = pick_point_from_1d_idx(raster=curr_y_grid, idx_1d=curr_1d_idx)
        sp_z_from_grid = pick_point_from_1d_idx(raster=curr_z_grid, idx_1d=curr_1d_idx)

        start_idx = get_2d_idx_from_1d_idx(indices=curr_1d_idx, shape=curr_z_grid.shape)
        start_coords = (sp_x_from_grid, sp_y_from_grid, sp_z_from_grid)  # Tuple for consistency
        
        logger.info(f"Generating path(s) for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y})")
        if method in ['gradient', 'd8-flow']:
            path_coords, path_idx, steps_count, stop_reason = generate_path(
                start_idx=start_idx, 
                start_coords=start_coords, 
                x_grid=curr_x_grid, 
                y_grid=curr_y_grid, 
                z_grid=curr_z_grid,
                method=method, 
                flow=flow_sense, 
                step_size=step_size, 
                max_steps=max_steps, 
                min_slope=min_slope_degrees, 
                invalid_steps_tolerance=invalid_steps_tolerance
            )
            if steps_count >= min_steps:
                paths_data.append({
                    'path_id': path_id,
                    'starting_point_id': sp_id,
                    'starting_coords': start_coords,
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
                start_idx, start_coords, curr_z_grid, curr_x_grid, curr_y_grid,
                flow_sense, step_size, max_steps, min_slope_degrees, invalid_steps_tolerance, min_steps
            )
            for path_coords, path_idx, steps_count, stop_reason in compatible_paths:
                paths_data.append({
                    'path_id': path_id,
                    'starting_point_id': sp_id,
                    'starting_coords': start_coords,
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