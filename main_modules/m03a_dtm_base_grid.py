# %% === Import necessary modules
import os
import sys
import argparse
import shapely
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    SUPPORTED_FILE_TYPES
)

# Importing necessary modules from psliptools
from psliptools.rasters import (
    load_georaster, 
    convert_coords_to_geo, 
    resample_raster, 
    plot_elevation_3d,
    mask_raster_with_1d_idx,
    get_1d_idx_from_2d_mask
)

from psliptools.utilities import (
    select_files_in_folder_prompt,
    select_dir_prompt
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger(__name__)
logger.info("=== Import DTM ===")

# %% === Helper functions
# Read and import DTM files in a dataframe
def import_dtm_files(
        env: AnalysisEnvironment,
        file_type: str, 
        files_path: list[str], 
        resample_size: tuple[int, int]=None, 
        resample_method: str='average',
        data_dtype: str=None,
        coord_dtype: str='float32',
        poly_mask: shapely.geometry.Polygon | shapely.geometry.MultiPolygon=None,
        apply_mask_to_raster: bool=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Import DTM files in a dataframe."""
    dtm_data = [] # Initializing an empty list
    abg_data = [] # Initializing an empty list
    cust_id = []
    for idx, dtm_path in enumerate(files_path):
        raster_data, raster_profile, raster_x, raster_y, mask_matrix = load_georaster(
            filepath=dtm_path,
            set_dtype=data_dtype, 
            convert_to_geo=False, 
            poly_mask=poly_mask,
            squeeze=True, # To remove the dimension with length 1
            set_coord_dtype=coord_dtype
        )

        if raster_data is None: # No pixels inside poly_load_mask_geo
            continue

        if raster_data.ndim != 2:
            raise ValueError(f"Raster data has {raster_data.ndim} dimensions, but expected 2.")

        _, curr_cust_id = env.add_input_file(file_path=dtm_path, file_type=file_type, file_subtype=f'dtm{idx+1}')
        if resample_size:
            raster_data, raster_profile, raster_x, raster_y, mask_matrix = resample_raster(
                in_raster=raster_data, 
                in_profile=raster_profile,
                resample_method=resample_method,
                new_size=resample_size,
                poly_mask=poly_mask
            )

        raster_lon, raster_lat = convert_coords_to_geo(
            crs_in=raster_profile['crs'].to_epsg(), 
            in_coords_x=raster_x, 
            in_coords_y=raster_y,
            force_ndarray=True
        )

        mask_idx_1d = get_1d_idx_from_2d_mask(mask_matrix, order='C') # C order is the default and means row-major (row by row)

        if apply_mask_to_raster:
            raster_data = mask_raster_with_1d_idx(raster_data, mask_idx_1d, profile=raster_profile)

        dtm_data.append({
            'file_id': curr_cust_id,
            'elevation': raster_data,
            'profile': raster_profile,
        })

        abg_data.append({
            'longitude': raster_lon,
            'latitude': raster_lat,
            'mask_idx_1d': mask_idx_1d
        })

        cust_id.append(curr_cust_id)
    
    dtm_df = pd.DataFrame(dtm_data) # Convert the list of dictionaries to a DataFrame
    abg_df = pd.DataFrame(abg_data) # Convert the list of dictionaries to a DataFrame

    return dtm_df, abg_df, cust_id

# %% === Main function to import DEM and define base grid
def main(
        base_dir: str=None, 
        gui_mode: bool=False, 
        resample_size: int=None, 
        resample_method: str='average',
        data_dtype: str=None,
        coord_dtype: str='float32',
        apply_mask_to_raster: bool=False, 
        check_plot: bool=False
    ) -> dict[str, object]:
    """Main function to define the base grid."""
    source_type = 'dtm'
    
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    study_area_cln_poly = env.load_variable(variable_filename='study_area_vars.pkl')['study_area_cln_poly']

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Directory selection ===")
        dtm_fold = select_dir_prompt(
            default_dir=env.folders['inputs'][source_type]['path'], 
            content_type=source_type
        )
        
        print("\n=== Raster selection ===")
        dtm_paths = select_files_in_folder_prompt(
            base_dir=dtm_fold, 
            src_ext=SUPPORTED_FILE_TYPES['raster'], 
            allow_multiple=True
        )
    
    dtm_df, abg_df, cust_id = import_dtm_files(
        env=env,
        file_type=source_type, 
        files_path=dtm_paths, 
        resample_size=resample_size, 
        resample_method=resample_method,
        data_dtype=data_dtype,
        coord_dtype=coord_dtype,
        poly_mask=study_area_cln_poly,
        apply_mask_to_raster=apply_mask_to_raster
    )

    dtm_abg_vars = {'dtm': dtm_df, 'abg': abg_df}

    env.config['inputs'][source_type][0]['settings'] = {
        'resample_size': resample_size, # None or (x, y)
        'resample_method': resample_method,
        'apply_mask_to_raster': apply_mask_to_raster
    }
    env.config['inputs'][source_type][0]['custom_id'] = cust_id
    env.collect_input_files(file_type=[source_type], multi_extension=True)

    env.save_variable(variable_to_save=dtm_abg_vars, variable_filename=f"{source_type}_vars.pkl", compression='gzip')

    # Check-plot
    if check_plot:
        plot_elevation_3d(dtm_df['elevation'], abg_df['longitude'], abg_df['latitude'], mask_idx_1d=abg_df['mask_idx_1d'], projected=True)
    
    return dtm_abg_vars

# %% === Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Define the base grid for the analysis, importing the DEM.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    parser.add_argument('--resample_size', type=int, default=None, help="Resample size for the DEM.")
    parser.add_argument('--resample_method', type=str, default='average', help="Resample method for the DEM.")
    parser.add_argument('--data_dtype', type=str, default=None, help="Raster data type for the DEM.")
    parser.add_argument('--coord_dtype', type=str, default='float32', help="Coordinate data type for the DEM.")
    parser.add_argument('--apply_mask_to_raster', action='store_true', help="Apply mask to raster data.")
    parser.add_argument('--check_plot', action='store_true', help="Check plot for the DEM.")
    args = parser.parse_args()

    dtm_abg_vars = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode, 
        resample_size=args.resample_size, 
        resample_method=args.resample_method,
        data_dtype=args.data_dtype,
        coord_dtype=args.coord_dtype,
        apply_mask_to_raster=args.apply_mask_to_raster,
        check_plot=args.check_plot
    )

# %%