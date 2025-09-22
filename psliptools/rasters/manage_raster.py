# %% === Import necessary modules
import numpy as np

# %% === Defaults
DEFAULT_ORDER_C = 'C' # C style (row-major)

# %% === Function to get 1d indices from mask
def get_1d_idx_from_2d_mask(
        mask: np.ndarray,
        order: str=DEFAULT_ORDER_C
    ) -> np.ndarray:
    """
    Get 1D indices from a 2D mask.
    
    Args:
        mask (np.ndarray): A 2D boolean array.
        order (str, optional): The order of the array ('C' -> C style, which means row-major; 
            or 'F' -> Fortran style, which means column-major). Defaults to 'C'.
        
    Returns:
        np.ndarray: A 1D array of indices.
    """
    if mask.ndim != 2:
        raise ValueError('mask must be a 2D array')
    
    idx_1d = np.where(mask.flatten(order=order))[0]
    return idx_1d

# %% === Function to obtain 1d indices of an array from 2d indices
def get_1d_idx_from_2d_idx(
        idx_2d: np.ndarray | tuple[np.ndarray, np.ndarray],
        shape: tuple, 
        order: str=DEFAULT_ORDER_C
    ) -> np.ndarray:
    """
    Get 1D indices from 2D indices.
    
    Args:
        idx_2d (np.ndarray | tuple): It can be specified in different ways:
            - A 2D array of indices (N x 2 or 2 x N, where N is the number of indices).
            - A tuple of two 1D arrays of indices (x, y).
        shape (tuple): The shape of the original 2D array.
        order (str, optional): The order of the array ('C' -> C style, which means row-major; 
            or 'F' -> Fortran style, which means column-major). Defaults to 'C'.
        
    Returns:
        np.ndarray: A 1D array of indices.
    """
    if isinstance(idx_2d, tuple):
        idx_2d = np.array(idx_2d)
    
    if idx_2d.ndim != 2:
        raise ValueError('idx_2d must be a 2D array')
    
    if idx_2d.shape[0] != 2:
        idx_2d = idx_2d.T
        if idx_2d.shape[0] != 2:
            raise ValueError('idx_2d must be a 2D array with 2 rows or two columns')
    
    idx_1d = np.ravel_multi_index(idx_2d, shape, order=order)
    return idx_1d

# %% === Function to obtain 2d indices of an array from 1d indices
def get_2d_idx_from_1d_idx(
        indices: np.ndarray, 
        shape: tuple, 
        order: str=DEFAULT_ORDER_C
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get 2D indices from 1D indices.
    
    Args:
        indices (np.ndarray): A 1D array of indices.
        shape (tuple): The shape of the 2D array.
        order (str, optional): The order that was used to create the array ('C' -> C style, which means row-major; 
            or 'F' -> Fortran style, which means column-major). Defaults to 'C'.
        
    Returns:
        tuple(np.ndarray, np.ndarray): Tuple containing the row and column indices.
    """
    if indices.ndim > 1:
        raise ValueError('indices must be a 1D array or scalar')
    
    # Check that all numbers are integer or can be converted
    if not issubclass(indices.dtype.type, np.integer):
        if np.isnan(indices).any():
            raise ValueError('NaN values are present in the array')
        if np.issubdtype(indices.dtype, np.floating):
            if np.allclose(indices % 1, 0):
                indices = indices.astype(int)
            else:
                raise ValueError('The array contains non-integer numbers')
        else:
            raise ValueError('The array contains non-numeric values')
    
    rows, cols = np.unravel_index(indices, shape, order=order)
    rows = np.squeeze(rows)
    cols = np.squeeze(cols)
    return (rows, cols)

# %% === Function to obtain 2d mask (True/False) from 1d indices
def get_2d_mask_from_1d_idx(
        indices: np.ndarray, 
        shape: tuple,
        order: str=DEFAULT_ORDER_C
    ) -> np.ndarray:
    """
    Get 2D mask from 1D indices.

    Args:
        indices (np.ndarray): A 1D array of indices.
        shape (tuple): The shape of the 2D array.
        order (str, optional): The order that was used to create the array ('C' -> C style, which means row-major; 
            or 'F' -> Fortran style, which means column-major). Defaults to 'C'.

    Returns:
        np.ndarray: A 2D boolean array.
    """
    mask = np.zeros(shape, dtype=bool)
    mask[get_2d_idx_from_1d_idx(indices, shape, order=order)] = True
    return mask

# %% === Function to mask a raster with a 1d idx array
def mask_raster_with_1d_idx(
        raster: np.ndarray, 
        mask_1d_idx: np.ndarray, 
        order: str=DEFAULT_ORDER_C,
        profile: dict=None
    ) -> np.ndarray:
    """
    Mask a raster with a 1D index array.

    Args:
        raster (np.ndarray): The raster to be masked.
        mask_1d_idx (np.ndarray): A 1D array of indices.
        order (str, optional): The order that was used to create the array ('C' -> C style, which means row-major; 
            or 'F' -> Fortran style, which means column-major). Defaults to 'C'.
        profile (dict, optional): The raster profile dictionary. Defaults to None.

    Returns:
        np.ndarray: The masked raster.
    """
    out_raster = raster.copy()
    mask = get_2d_mask_from_1d_idx(mask_1d_idx, out_raster.shape, order=order)
    excluded = np.logical_not(mask)
    if profile is None or profile.get('nodata', None) is None:
        out_raster[excluded] = 0 # Default nodata value set to 0
    else:
        out_raster[excluded] = profile['nodata']
    return out_raster

# %% === Function to pick a point from a raster with a 1d index
def pick_point_from_1d_idx(
        raster: np.ndarray, 
        idx_1d: np.ndarray,
        order: str=DEFAULT_ORDER_C
    ) -> np.ndarray:
    """
    Pick a point from a raster with a 1D index array.

    Args:
        raster (np.ndarray): The raster to be picked from.
        idx_1d (np.ndarray): A 1D array of indices.
        order (str, optional): The order that was used to create the array ('C' -> C style, which means row-major; 
            or 'F' -> Fortran style, which means column-major). Defaults to 'C'.

    Returns:
        np.ndarray: The picked point(s).
    """
    rows, cols = get_2d_idx_from_1d_idx(indices=idx_1d, shape=raster.shape, order=order)
    values = raster[rows, cols]
    return values

# %%
