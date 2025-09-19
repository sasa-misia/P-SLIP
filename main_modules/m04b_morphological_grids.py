# %% === Import necessary modules
import os
import sys
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment,
    LOG_CONFIG
)

# # Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

# from psliptools.utilities import (
# )

# from psliptools.geometries import (
# )

# Importing necessary modules from main_modules
from env_init import get_or_create_analysis_environment

# %% === Set up logging configuration
# This will log messages to the console and can be modified to log to a file if needed
logging.basicConfig(level=logging.INFO,
                    format=LOG_CONFIG['format'], 
                    datefmt=LOG_CONFIG['date_format'])

# %% === Methods to create morphological grids

# %% === Main function
def main(
        gui_mode: bool=False, 
        base_dir: str=None
    ) -> None:
    """Main function to create morphological grids"""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    # TODO: Add code to create morphological grids (slope, aspect, curvatures, etc.)

# %% === Command line interface