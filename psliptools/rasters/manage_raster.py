# %% === Import necessary modules
import numpy as np
import pandas as pd

from .coordinates import generate_fake_xy_grids

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
        indices: np.ndarray | list[int | float] | int | float, 
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
    indices = np.array(indices)

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
        idx_1d: np.ndarray | list[int | float] | int | float,
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
    if not isinstance(raster, np.ndarray):
        raise ValueError('raster must be a numpy array')
    
    rows, cols = get_2d_idx_from_1d_idx(indices=idx_1d, shape=raster.shape, order=order)
    values = raster[rows, cols]
    return values

# %% === Function to get D8 neighbors
def get_d8_neighbors_row_col(
        row: int | float | np.ndarray,
        col: int | float | np.ndarray, 
        grid_shape: tuple[int, int],
        search_size: int=1
    ) -> np.ndarray:
    """
    Get 8 possible neighbors (D8 directions, clockwise starting from east).

    Args:
        row (int): The row index of the center pixel.
        col (int): The column index of the center pixel.
        grid_shape (tuple): The shape of the grid (rows, columns).
        search_size (int, optional): The size of the search window, which means the number of pixels to search around the center pixel (default is 1).

    Returns:
        np.ndarray: A 2D array of neighbors indices (row, col) (-1 indicates no neighbor because it is outside the grid).
    """
    row = np.atleast_1d(row)
    col = np.atleast_1d(col)
    grid_shape = np.atleast_1d(grid_shape)
    search_size = np.atleast_1d(search_size)
    if row.size != 1 or row%1 != 0:
        raise ValueError('row must be integer and scalar')
    if col.size != 1 or col%1 != 0:
        raise ValueError('col must be integer and scalar')
    if grid_shape.size != 2 or grid_shape[0]%1 != 0 or grid_shape[1]%1 != 0:
        raise ValueError('grid_shape must be a tuple of integer with length 2')
    if search_size.size != 1 or search_size%1 != 0:
        raise ValueError('search_size must be integer and scalar')
    
    row = row.item()
    col = col.item()
    grid_shape = grid_shape.astype(int)
    search_size = search_size.item()
    
    directions = np.array( # D8 directions clockwise starting from east
        [
            (0, search_size), # Right (E)
            (search_size, search_size), # Bottom-right (SE)
            (search_size, 0), # Bottom (S)
            (search_size, -search_size), # Bottom-left (SW)
            (0, -search_size), # Left (W)
            (-search_size, -search_size), # Top-left (NW)
            (-search_size, 0), # Top (N)
            (-search_size, search_size), # Top-right (NE)
        ],
        dtype=int
    )

    neighbors = np.array([row, col]) + directions
    valid_mask = np.array([
        (0 <= neighbors[:,0]) & (neighbors[:,0] < grid_shape[0]),
        (0 <= neighbors[:,1]) & (neighbors[:,1] < grid_shape[1])
    ]).T
    neighbors[~valid_mask] = -1
    
    return neighbors

# %% Function to get the slope of the D8 neighbors
def get_d8_neighbors_slope(
        row: int | float | np.ndarray,
        col: int | float | np.ndarray,
        z_grid: np.ndarray,
        x_grid: np.ndarray=None,
        y_grid: np.ndarray=None,
        search_size: int=1,
        output_format: str='pandas' # or 'numpy'
    ) -> pd.DataFrame | np.ndarray:
    """
    Get the slope of the D8 neighbors.

    Args:
        row (int): The row index of the center pixel.
        col (int): The column index of the center pixel.
        z_grid (np.ndarray): The elevation data.
        x_grid (np.ndarray, optional): The x coordinates of the grid (default is None).
        y_grid (np.ndarray, optional): The y coordinates of the grid (default is None).
        search_size (int, optional): The size of the search window, which means the number of pixels to search around the center pixel (default is 1).
        output_format (str, optional): The output format ('pandas' or 'numpy') (default is 'pandas'). 
            - if pandas, returns a DataFrame with multiple info about the neighbors;
            - if numpy, returns just the array of the D8 neighbors slopes.

    Returns:
        (pd.DataFrame | np.ndarray): The slope of the D8 neighbors (slope is just the elevation difference divided by the distance between the current pixel and the neighbor, not the slope in degrees!).
    """
    if x_grid is None or y_grid is None:
        x_grid, y_grid = generate_fake_xy_grids(z_grid.shape) # Fake base grid with size 1x1
    
    center_coords = np.array((x_grid[row, col], y_grid[row, col], z_grid[row, col]))
    neighbors = get_d8_neighbors_row_col(row, col, z_grid.shape, search_size=search_size)

    edge_mask = (neighbors == -1).any(axis=1)

    neighbor_coords = np.array([
        x_grid[neighbors[:,0], neighbors[:,1]], 
        y_grid[neighbors[:,0], neighbors[:,1]], 
        z_grid[neighbors[:,0], neighbors[:,1]]
    ]).T
    plane_length = np.sqrt(np.sum((neighbor_coords[:, 0:2] - center_coords[0:2])**2, axis=1))
    tridim_length = np.sqrt(np.sum((neighbor_coords - center_coords)**2, axis=1))
    delta_z = neighbor_coords[:,2] - center_coords[2] # positive value means uphill
    slopes = delta_z / plane_length

    neighbor_coords[edge_mask] = np.nan
    plane_length[edge_mask] = np.nan
    tridim_length[edge_mask] = np.nan
    delta_z[edge_mask] = np.nan
    slopes[edge_mask] = np.nan
    
    if output_format == 'pandas':
        row_indices = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
        slopes_df = pd.DataFrame(
            {
                'row_start': row,
                'col_start': col,
                'row_end': neighbors[:,0],
                'col_end': neighbors[:,1],
                'coords_start': [center_coords],
                'coords_end': neighbor_coords.tolist(),
                'plane_length': plane_length,
                'tridim_length': tridim_length,
                'delta_z': delta_z,
                'slope': slopes,
            },
            index=row_indices
        )

        slopes_out = slopes_df
    elif output_format == 'numpy':
        slopes_out = slopes
    else:
        raise ValueError('output_format must be either "pandas" or "numpy"')
    
    return slopes_out

# %% === Function to get gradients at a certain point of a grid
def get_point_gradients(
        row: int | float | np.ndarray,
        col: int | float | np.ndarray,
        z_grid: np.ndarray,
        x_grid: np.ndarray=None,
        y_grid: np.ndarray=None,
        search_size: int=1
    ) -> tuple[float, float]:
    """
    Calculate gradient vector (dx, dy) at current position using finite differences.

    Args:
        row (int): The row index of the center pixel.
        col (int): The column index of the center pixel.
        z_grid (np.ndarray): The elevation data.
        x_grid (np.ndarray, optional): The x coordinates of the grid (default is None).
        y_grid (np.ndarray, optional): The y coordinates of the grid (default is None).
        search_size (int, optional): The size of window to use for finite differences, 
            which means that the gradient is calculated using the pixels that are 
            search_size away from the current pixel (default is 1).
    
    Returns:
        tuple[float, float]: The gradient vector (dx, dy). If the box for the gradient is outside the grid, returns nans in one or both directions.
    """
    if x_grid is None or y_grid is None:
        x_grid, y_grid = generate_fake_xy_grids(z_grid.shape) # Fake base grid with size 1x1
    
    rows, cols = z_grid.shape
    search_size = np.int64(search_size)
    row_d = row + search_size # Down pixel
    row_u = row - search_size # Up pixel
    col_l = col - search_size # Left pixel
    col_r = col + search_size # Right pixel

    if col_l > 0 and col_r < cols - 1:
        z_left  = z_grid[row, col_l]
        z_right = z_grid[row, col_r]
        x_left  = x_grid[row, col_l]
        x_right = x_grid[row, col_r]
        grad_x  = (z_right - z_left) / np.abs(x_right - x_left)
    else:
        grad_x = float('nan')  # Edge, no gradient

    if row_u > 0 and row_d < rows - 1:
        z_up   = z_grid[row_u, col]
        z_down = z_grid[row_d, col]
        y_up   = y_grid[row_u, col]
        y_down = y_grid[row_d, col]
        grad_y = (z_down - z_up) / np.abs(y_down - y_up) # First down because has greater index
    else:
        grad_y = float('nan') # Edge, no gradient
    
    return grad_x, grad_y

# %%
