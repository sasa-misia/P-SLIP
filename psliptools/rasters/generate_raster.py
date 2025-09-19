# %% === Import necessary modules
import os
import warnings
import numpy as np
import pandas as pd

# %% === Method to create a raster grid of a specific parameter
def generate_grids_from_indices(
        shapes: list[tuple[int, int]],
        indices: list[list[np.ndarray]],
        csv_paths: list[str],
        class_names: list[str],
        parameter_name: str,
        out_type: str='float32',
        no_data: float=0
    ) -> np.ndarray:
    """Generate a raster grid from a P-SLIP dataframe of indices and the associated AnalysisBaseGrid."""
    # TODO: Create a function to create a raster grid from a P-SLIP dataframe of indices and the associated AnalysisBaseGrid.
