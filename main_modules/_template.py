# %% === Import necessary modules
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment
)

# # Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

# from psliptools.utilities import (
# )

# from psliptools.geometries import (
# )

# from psliptools.scattered import (
# )

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger(__name__)
logger.info("=== Module ===")

# %% === Helper functions

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False
    ) -> None:
    """Main function to ..."""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

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