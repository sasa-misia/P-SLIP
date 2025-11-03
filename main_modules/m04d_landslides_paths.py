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
    get_d8_neighbors_slope,
    get_point_gradients,
    get_closest_1d_pixel_idx,
    convert_coords
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

def generate_steepsets_path(
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
    ) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Generate a single path from start point."""
    path_coord_list = [start_coords]
    path_2D_idx_list = [start_idx]
    steps_count = 0
    invalid_steps = 0
    stop_reason = None
    curr_row, curr_col = start_idx

    dx = np.abs(np.mean(x_grid[curr_row, 1:] - x_grid[curr_row, :-1]))
    dy = np.abs(np.mean(y_grid[1:, curr_col] - y_grid[:-1, curr_col]))

    curr_x, curr_y = start_coords[0].copy(), start_coords[1].copy()
    
    while steps_count < max_steps:
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
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
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

            if grad_magnitude < min_slope:  # Use gradient magnitude as slope proxy
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
        if (next_row, next_col) in path_2D_idx_list:
            stop_reason = 'loop_detected'
            break
        
        # Check slope tolerance
        if invalid_steps > invalid_steps_tolerance:
            stop_reason = 'invalid_steps_tolerance_exceeded'
            break
        
        # Move to next point
        curr_row, curr_col = next_row, next_col
        path_coord_list.append((x_grid[curr_row, curr_col], y_grid[curr_row, curr_col], z_grid[curr_row, curr_col]))
        path_2D_idx_list.append((curr_row, curr_col))
        steps_count += 1
    
    # Check max runout
    if steps_count >= max_steps:
        stop_reason = 'max_runout'

    if stop_reason is None:
        stop_reason = 'completed'

    path_coords = np.array(path_coord_list)
    path_2D_idx = np.array(path_2D_idx_list)
    
    return path_coords, path_2D_idx, steps_count, stop_reason

def generate_slope_compatible_paths(start_idx, start_coords, grid, lon_grid, lat_grid, flow_sense, step_size, max_steps, min_slope_degrees, invalid_steps_tolerance, min_steps):
    """Generate multiple slope-compatible paths (recursive branching)."""
    # TODO: Check this function!
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

    abg_prj_df, starting_points_prj_df, prj_epsg_code = convert_abg_and_ref_points_to_prj( # Both DataFrames are now projected and contain prj_x and prj_y columns
        abg_df=abg_df,
        ref_points_df=starting_points_df
    )
    
    # Generate paths
    logger.info("Generating paths...")
    generated_paths = []
    path_count = 0
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
        sp_x_from_grid = pick_point_from_1d_idx(raster=curr_x_grid, idx_1d=curr_1d_idx)
        sp_y_from_grid = pick_point_from_1d_idx(raster=curr_y_grid, idx_1d=curr_1d_idx)
        sp_z_from_grid = pick_point_from_1d_idx(raster=curr_z_grid, idx_1d=curr_1d_idx)

        start_idx = get_2d_idx_from_1d_idx(indices=curr_1d_idx, shape=curr_z_grid.shape)
        start_prj_coords = (sp_x_from_grid, sp_y_from_grid, sp_z_from_grid)  # Tuple for consistency
        
        logger.info(f"Generating path(s) for starting point [id: {sp_id}] (x: {sp_x}, y: {sp_y})")
        if method in ['gradient', 'd8-flow']:
            path_prj_coords, path_idx, steps_count, stop_reason = generate_steepsets_path(
                start_idx=start_idx, 
                start_coords=start_prj_coords, 
                x_grid=curr_x_grid, 
                y_grid=curr_y_grid, 
                z_grid=curr_z_grid,
                method=method, 
                flow=flow_sense, 
                step_size=step_size, 
                max_steps=max_steps, 
                min_slope=np.tan(np.deg2rad(min_slope_degrees)), 
                invalid_steps_tolerance=invalid_steps_tolerance
            )
            compatible_paths = [(path_prj_coords, path_idx, steps_count, stop_reason)]
        elif method == 'slope-compatible': # TODO: Check this method!
            compatible_paths = generate_slope_compatible_paths(
                start_idx, start_prj_coords, curr_z_grid, curr_x_grid, curr_y_grid,
                flow_sense, step_size, max_steps, min_slope_degrees, invalid_steps_tolerance, min_steps
            )
        
        for path_prj_coords, path_idx, steps_count, stop_reason in compatible_paths:
            if steps_count >= min_steps:
                path_lon, path_lat = convert_coords(crs_in=prj_epsg_code, crs_out=4326, in_coords_x=path_prj_coords[:, 0], in_coords_y=path_prj_coords[:, 1])
                path_geo_coords = path_prj_coords.copy()
                path_geo_coords[:, 0] = path_lon
                path_geo_coords[:, 1] = path_lat
                generated_paths.append({
                    'path_id': f"PLP_{path_count}", # Potential Landslide Path PLP
                    'starting_point_id': sp_id,
                    'starting_source': starting_source,
                    'method': method,
                    'flow_sense': flow_sense,
                    'steps_count': steps_count,
                    'stop_reason': stop_reason,
                    'path_dtm': curr_dtm,
                    'start_snap_offset_meters': curr_dist_to_grid_point,
                    'path_2D_idx': path_idx,
                    'path_geo_coords': path_geo_coords
                })
                path_count += 1
    
    landslide_paths_vars = {'paths_df': pd.DataFrame(generated_paths)}
    return landslide_paths_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="**********Summary of the module**********")
    parser.add_argument("--base_dir", type=str, help="Base directory for analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode")
    parser.add_argument("--method", type=str, default='gradient', help="Path generation method (gradient, d8-flow, slope-compatible)")
    parser.add_argument("--flow_sense", type=str, default='upstream', help="Flow sense (upstream, downstream)")
    parser.add_argument("--step_size", type=float, default=1, help="Step size in meters")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps for the generated path")
    parser.add_argument("--min_steps", type=int, default=2, help="Minimum number of steps for the generated path")
    parser.add_argument("--min_slope_degrees", type=float, default=5, help="Minimum slope in degrees for the generated path")
    parser.add_argument("--invalid_steps_tolerance", type=float, default=0.5, help="Tolerance for invalid steps in meters")
    
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
        invalid_steps_tolerance=args.invalid_steps_tolerance
    )