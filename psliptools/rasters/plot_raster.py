# %% === Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import mayavi.mlab as mlab
from .coordinates import convert_grids_and_profile_to_prj
from .manage_raster import mask_raster_with_1d_idx, get_2d_mask_from_1d_idx

# %% === Function to create a fake base grid
def _create_fake_base_grid(
        elevation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a fake base grid.
    
    Args:
        elevation (np.ndarray): The elevation data.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: The x and y coordinates of the fake base grid.
    """
    x_grid, y_grid = np.meshgrid(
        np.arange(1, elevation.shape[-1] + 1), 
        np.arange(elevation.shape[-2], 0, -1)
    )
    return x_grid, y_grid

# %% === Function to regularize elevation
def _regularize_elevation(
        elevation: np.ndarray, 
        x_grid: np.ndarray=None, 
        y_grid: np.ndarray=None
    ) -> np.ndarray:
    """
    Regularize the elevation data.
    
    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        
    Returns:
        np.ndarray: The regularized elevation data.
    """
    out_elevation = elevation.copy() # Create a copy to avoid modifying the original data
    if out_elevation.ndim == 3:
        out_elevation = out_elevation[0] # Only use the first layer
        warnings.warn('elevation is a 3D array, only the first layer will be used')
    elif out_elevation.ndim == 1 and x_grid is not None and y_grid is not None:
        out_elevation = out_elevation.reshape(x_grid.shape)
    
    if not out_elevation.ndim == 2:
        raise ValueError('elevation must be a 2D array')
    return out_elevation

# %% === Function to check consistency between elevation, x_grid and y_grid
def _check_grid_consistency(
        elevation: np.ndarray, 
        x_grid: np.ndarray, 
        y_grid: np.ndarray,
        mask2d: np.ndarray=None
    ) -> None:
    """
    Check consistency between elevation, x_grid and y_grid.
    
    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        mask2d (np.ndarray, optional): The 2D mask (True for points to show, False for points to hide).
        
    Returns:
        None
        
    Raises:
        ValueError: If x_grid and y_grid do not have the same shape as elevation.
    """
    if x_grid.shape != y_grid.shape:
        raise ValueError('x_grid and y_grid must have the same shape')
    if x_grid.shape != elevation.shape:
        raise ValueError('x_grid/y_grid and elevation must have the same shape')
    if mask2d is not None and mask2d.shape != elevation.shape:
        raise ValueError('mask2d must have the same shape as elevation')
    
# %% === Function to regularize grids
def _regularize_zxy(
        elevation: np.ndarray, 
        x_grid: np.ndarray=None, 
        y_grid: np.ndarray=None,
        mask_idx_1d: np.ndarray=None,
        projected: bool=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regularize x_grid, y_grid and elevation.
    
    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The regularized elevation, x_grid, y_grid and mask_idx_1d.
    """
    elevation = _regularize_elevation(elevation, x_grid, y_grid)
    if x_grid is None or y_grid is None:
        warnings.warn('x_grid and/or y_grid are not provided, creating a fake base grid...')
        x_grid, y_grid = _create_fake_base_grid(elevation)
    else:
        if projected:
            x_grid, y_grid, _ = convert_grids_and_profile_to_prj(x_grid, y_grid)
    
    if mask_idx_1d is None:
        mask_idx_1d = np.arange(x_grid.size)
    else:
        mask_idx_1d = np.array(mask_idx_1d)
    
    mask2d = get_2d_mask_from_1d_idx(mask_idx_1d, x_grid.shape)
    if mask2d is not None and not issubclass(mask2d.dtype.type, np.bool_):
        raise ValueError('mask2d must be a boolean array')

    _check_grid_consistency(elevation, x_grid, y_grid, mask2d)
    return elevation, x_grid, y_grid, mask_idx_1d

# %% === Function to regularize arrays to lists
def _regularize_zxy_multi(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
        mask_idx_1d: list | pd.Series | np.ndarray=None,
        projected: bool=False
    ) -> tuple[list, list, list]:
    """
    Regularize elevation, x_grid, and y_grid.
    
    Args:
        elevation (list | pd.Series | np.ndarray): The elevation data.
        x_grid (list | pd.Series | np.ndarray, optional): The x coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        y_grid (list | pd.Series | np.ndarray, optional): The y coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        mask_idx_1d (list | pd.Series | np.ndarray, optional): The indices of the points to show (1d indices).
        projected (bool, optional): Whether the grid is projected.
        
    Returns:
        tuple[list, list, list, list]: The regularized elevation, x_grid, y_grid and mask_idx_1d.
    """
    if isinstance(elevation, np.ndarray): # in this case elevation is a single matrix
        elevation, x_grid, y_grid, mask_idx_1d = _regularize_zxy(elevation, x_grid, y_grid, mask_idx_1d, projected)
        elevation = [elevation]
        x_grid = [x_grid]
        y_grid = [y_grid]
        mask_idx_1d = [mask_idx_1d]
    elif x_grid is None or y_grid is None:
        raise ValueError('x_grid and y_grid must be provided (as lists) when elevation is split into multiple matrices')
    
    if isinstance(elevation, pd.Series) or isinstance(elevation, pd.DataFrame):
        elevation = elevation.to_list()
        x_grid = x_grid.to_list()
        y_grid = y_grid.to_list()
        mask_idx_1d = mask_idx_1d.to_list()
    
    if not(isinstance(elevation, list) and isinstance(x_grid, list) and isinstance(y_grid, list) and isinstance(mask_idx_1d, list)):
        raise ValueError(f"elevation, x_grid, y_grid, and mask_idx_1d must be lists, not {type(elevation)}, {type(x_grid)}, {type(y_grid)}")
    
    if not(len(elevation) == len(x_grid) == len(y_grid) == len(mask_idx_1d)):
        raise ValueError(f"elevation, x_grid, y_grid, and mask_idx_1d must have the same length, not {len(elevation)}, {len(x_grid)}, {len(y_grid)}")
    
    for idx, (e, x, y, m) in enumerate(zip(elevation, x_grid, y_grid, mask_idx_1d)):
        elevation[idx], x_grid[idx], y_grid[idx], mask_idx_1d[idx] = _regularize_zxy(e, x, y, m, projected)
    return elevation, x_grid, y_grid, mask_idx_1d

# %% === Function to obtain the masked raster
def _obtain_masked_raster(
        elevation: list, 
        mask_idx_1d: list
    ) -> list:
    """
    Obtain the masked raster.
    
    Args:
        elevation (list): List of the elevation data rasters.
        mask_idx_1d (list): List of the indices of the points to show for each raster (1d arrays with C order = row by row).
        
    Returns:
        list: List of the masked rasters.
    """
    if not isinstance(elevation, list) or not isinstance(mask_idx_1d, list):
        raise ValueError(f"elevation and mask_idx_1d must be lists, not {type(elevation)}, {type(mask_idx_1d)}")
    
    if len(elevation) != len(mask_idx_1d):
        raise ValueError(f"elevation and mask_idx_1d must have the same length, not {len(elevation)}, {len(mask_idx_1d)}")
    
    out_elevation = []
    for e, m in zip(elevation, mask_idx_1d):
        out_elevation.append(mask_raster_with_1d_idx(e, m))
    return out_elevation

# %% === Function to show elevation in a isometric plot
def plot_elevation_isometric(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
        mask_idx_1d: list | pd.Series | np.ndarray=None,
        figure: plt.figure=None,
        show: bool=True
    ) -> tuple[plt.figure, plt.axes]:
    """
    Show elevation in a isometric plot.
    
    Args:
        elevation (list | pd.Series | np.ndarray): The elevation data, which can be a single matrix or a list of matrices.
        x_grid (list | pd.Series | np.ndarray): The x coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        y_grid (list | pd.Series | np.ndarray): The y coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        mask_idx_1d (list | pd.Series | np.ndarray): The indices of the points to show (1d indices).
        figure (plt.figure): The figure to plot on.
        show (bool): Whether to show the plot.
        
    Returns:
        tuple[plt.figure, plt.axes]: The figure and axes.
    """
    elevation, x_grid, y_grid, mask_idx_1d = _regularize_zxy_multi(elevation, x_grid, y_grid, mask_idx_1d) # If mask_idx_1d was None, now it is an array with all True values
    elevation = _obtain_masked_raster(elevation, mask_idx_1d) 

    if figure is None:
        fig = plt.figure()
    else:
        fig = figure

    axs = fig.add_subplot(111, projection='3d', aspect='equal')
    axs.set_title('Elevation Map')
    for e, x, y in zip(elevation, x_grid, y_grid):
        axs.plot_surface(x, y, e, cmap='viridis')

    if show: fig.show()
    return fig, axs

# %% === Function to show elevation in a 2D plot
def plot_elevation_2d(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
        mask_idx_1d: list | pd.Series | np.ndarray=None,
        figure: plt.figure=None,
        show: bool=True
    ) -> tuple[plt.figure, plt.axes]:
    """
    Show elevation in a 2D plot.

    Args:
        elevation (list | pd.Series | np.ndarray): The elevation data, which can be a single matrix or a list of matrices.
        x_grid (list | pd.Series | np.ndarray): The x coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        y_grid (list | pd.Series | np.ndarray): The y coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        mask_idx_1d (np.ndarray): The indices of the points to show.
        figure (plt.figure): The figure to plot on.
        show (bool): Whether to show the plot.
        
    Returns:
        tuple[plt.figure, plt.axes]: The figure and axes.
    """
    elevation, x_grid, y_grid, mask_idx_1d = _regularize_zxy_multi(elevation, x_grid, y_grid, mask_idx_1d) # If mask_idx_1d was None, now it is an array with all True values
    elevation = _obtain_masked_raster(elevation, mask_idx_1d) 

    if figure is None:
        fig, axs = plt.subplots(nrows=1, ncols=len(elevation), figsize=(5 * len(elevation), 5))
        if len(elevation) == 1:
            axs = [axs]  # Ensure axes is always a list
    else:
        fig = figure
        axs = [fig.add_subplot(1, len(elevation), i + 1) for i in range(len(elevation))]

    for idx, (ax, e, x, y) in enumerate(zip(axs, elevation, x_grid, y_grid)):
        ax.set_title(f'Elevation Map {idx + 1}')
        ax.imshow(e, cmap='viridis', extent=(x[0, 0], x[-1, -1], y[0, 0], y[-1, -1]))
        ax.set_aspect('equal')

    if show: fig.show()
    return fig, axs

# %% === Function to show elevation in a 3D interactive plot
def plot_elevation_3d(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
        mask_idx_1d: list | pd.Series | np.ndarray=None,
        projected: bool=False,
        figure: mlab.figure=None,
        show: bool=True
    ) -> tuple[mlab.figure, any]:
    """
    Show elevation in a 3D interactive plot.

    Args:
        elevation (list | pd.Series | np.ndarray): The elevation data.
        x_grid (list | pd.Series | np.ndarray, optional): The x coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        y_grid (list | pd.Series | np.ndarray, optional): The y coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        mask_idx_1d (np.ndarray, optional): The indices of the points to show.
        projected (bool, optional): If True, the x_grid and y_grid will be converted to projected coordinates.
        figure (mlab.figure, optional): The figure to plot on.
        show (bool, optional): Whether to show the plot.

    Returns:
        tuple[mlab.figure, mlab.axes]: The figure and axes.
    """
    elevation, x_grid, y_grid, mask_idx_1d = _regularize_zxy_multi(elevation, x_grid, y_grid, mask_idx_1d, projected) # If mask_idx_1d was None, now it is an array with all True values
    elevation = _obtain_masked_raster(elevation, mask_idx_1d) 

    # create a unique color bar for all dtms
    min_elv = min([e.min() for e in elevation])
    max_elv = max([e.max() for e in elevation])

    if figure is None:
        fig = mlab.figure()
    else:
        fig = figure
    # axs = mlab.axes(figure=fig)
    axs = None
    
    for e, x, y in zip(elevation, x_grid, y_grid):
        mlab.mesh(x, y, e, colormap='terrain', vmin=min_elv, vmax=max_elv, figure=fig)

    if show: mlab.show()
    return fig, axs
    
# %% ===
