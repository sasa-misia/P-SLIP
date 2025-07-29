# %% === Import necessary modules
import numpy as np

# %% === Function to obtain 2d indices of an array from 1d indices
def get_2d_idx_from_1d_idx(
        indices: np.ndarray, 
        shape: tuple, 
        order: str='C'
    ) -> np.ndarray:
    rows, cols = np.unravel_index(indices, shape, order=order)
    rows = np.squeeze(rows)
    cols = np.squeeze(cols)
    return (rows, cols)

# %% === Function to obtain 2d mask (True/False) from 1d indices
def get_2d_mask_from_1d_idx(
        indices: np.ndarray, 
        shape: tuple,
        order: str='C'
    ) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[get_2d_idx_from_1d_idx(indices, shape, order=order)] = True
    return mask

# %% === Function to mask a raster with a 1d idx array
def mask_raster_with_1d_idx(
        raster: np.ndarray, 
        mask_1d_idx: np.ndarray, 
        order: str='C',
        profile: dict=None
    ) -> np.ndarray:
    out_raster = raster.copy()
    mask = get_2d_mask_from_1d_idx(mask_1d_idx, out_raster.shape, order=order)
    excluded = np.logical_not(mask)
    if profile is None or profile.get('nodata', None) is None:
        out_raster[excluded] = 0 # Default nodata value set to 0
    else:
        out_raster[excluded] = profile['nodata']
    return out_raster

# %% ===