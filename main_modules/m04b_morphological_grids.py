# %% === Import necessary modules
import os
import sys
import pandas as pd
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
# from config import (
# )

# Importing necessary modules from psliptools
from psliptools.rasters import (
    generate_slope_and_aspect_rasters,
    generate_curvature_rasters
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger()
logger.info("=== Create morphological grids ===")

# %% === Methods to create morphological grids
def get_angles_and_curvatures(
        abg_df: pd.DataFrame,
        dtm_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get angles and curvatures dataframes"""
    angles_dict = []
    curvatures_dict = []
    for (_, abg_row), (_, dtm_row) in zip(abg_df.iterrows(), dtm_df.iterrows()):
        curr_slope, curr_aspect = generate_slope_and_aspect_rasters(
            dtm=dtm_row['elevation'], 
            lon=abg_row['longitude'], 
            lat=abg_row['latitude'],
            out_type='float32',
            no_data=dtm_row['raster_profile']['nodata']
        )

        curr_prof, curr_plan, curr_twist = generate_curvature_rasters(
            dtm=dtm_row['elevation'], 
            lon=abg_row['longitude'], 
            lat=abg_row['latitude'],
            out_type='float32',
            no_data=dtm_row['raster_profile']['nodata']
        )

        angles_dict.append({
            'file_id': dtm_row['file_id'],
            'slope': curr_slope,
            'aspect': curr_aspect,
        })

        curvatures_dict.append({
            'file_id': dtm_row['file_id'],
            'profile': curr_prof,
            'planform': curr_plan,
            'twisting': curr_twist
        })
    
    angles_df = pd.DataFrame(angles_dict)
    curvatures_df = pd.DataFrame(curvatures_dict)

    return angles_df, curvatures_df

# %% === Main function
def main(
        gui_mode: bool=False, 
        base_dir: str=None
    ) -> None:
    """Main function to create morphological grids"""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    dtm_vars = env.load_variable(variable_filename='dtm_vars.pkl')

    abg_df = dtm_vars['abg']
    dtm_df = dtm_vars['dtm']

    logger.info("Getting angles and curvatures of each dtm stored in dtm_df...")
    angles_df, curvatures_df = get_angles_and_curvatures(abg_df, dtm_df)

    morphology_vars = {
        'angles': angles_df,
        'curvatures': curvatures_df
    }

    env.save_variable(variable_to_save=morphology_vars, variable_filename='morphology_vars.pkl')

    return morphology_vars

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create morphological grids.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    args = parser.parse_args()

    morphology_vars = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode
    )