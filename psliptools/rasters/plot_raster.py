#%% # Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import mayavi.mlab as mlab
from .coordinates import convert_grids_and_profile_to_prj

#%% # Function to create a fake base grid
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

#%% # Function to regularize elevation
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

#%% # Function to check consistency between elevation, x_grid and y_grid
def _check_grid_consistency(
        elevation: np.ndarray, 
        x_grid: np.ndarray, 
        y_grid: np.ndarray
    ) -> None:
    """
    Check consistency between elevation, x_grid and y_grid.
    
    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        
    Returns:
        None
        
    Raises:
        ValueError: If x_grid and y_grid do not have the same shape as elevation.
    """
    if x_grid.shape != y_grid.shape:
        raise ValueError('x_grid and y_grid must have the same shape')
    if x_grid.shape != elevation.shape:
        raise ValueError('x_grid/y_grid and elevation must have the same shape')
    
#%% # Function to regularize grids
def _regularize_zxy(
        elevation: np.ndarray, 
        x_grid: np.ndarray=None, 
        y_grid: np.ndarray=None,
        projected: bool=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regularize x_grid, y_grid and elevation.
    
    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The regularized x_grid, y_grid and elevation.
    """
    elevation = _regularize_elevation(elevation, x_grid, y_grid)
    if x_grid is None or y_grid is None:
        warnings.warn('x_grid and/or y_grid are not provided, creating a fake base grid...')
        x_grid, y_grid = _create_fake_base_grid(elevation)
    else:
        if projected:
            x_grid, y_grid, _ = convert_grids_and_profile_to_prj(x_grid, y_grid)
    _check_grid_consistency(elevation, x_grid, y_grid)
    return elevation, x_grid, y_grid

#%% # Function to regularize arrays to lists
def _regularize_zxy_multi(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
        projected: bool=False
    ) -> tuple[list, list, list]:
    """
    Regularize elevation, x_grid, and y_grid.
    
    Args:
        elevation (list | pd.Series | np.ndarray): The elevation data.
        x_grid (list | pd.Series | np.ndarray, optional): The x coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        y_grid (list | pd.Series | np.ndarray, optional): The y coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        
    Returns:
        tuple[list, list, list]: The regularized x_grid, y_grid and elevation.
    """
    if isinstance(elevation, np.ndarray): # in this case elevation is a single matrix
        elevation, x_grid, y_grid = _regularize_zxy(elevation, x_grid, y_grid, projected)
        elevation = [elevation]
        x_grid = [x_grid]
        y_grid = [y_grid]
    elif x_grid is None or y_grid is None:
        raise ValueError('x_grid and y_grid must be provided (as lists) when elevation is split into multiple matrices')
    
    if isinstance(elevation, pd.Series) or isinstance(elevation, pd.DataFrame):
        elevation = elevation.to_list()
        x_grid = x_grid.to_list()
        y_grid = y_grid.to_list()
    
    if not(isinstance(elevation, list) and isinstance(x_grid, list) and isinstance(y_grid, list)):
        raise ValueError(f"elevation, x_grid and y_grid must be lists, not {type(elevation)}, {type(x_grid)}, {type(y_grid)}")
    
    if len(elevation) != len(x_grid) or len(x_grid) != len(y_grid):
        raise ValueError(f"elevation, x_grid and y_grid must have the same length, not {len(elevation)}, {len(x_grid)}, {len(y_grid)}")
    
    for idx, (e, x, y) in enumerate(zip(elevation, x_grid, y_grid)):
        elevation[idx], x_grid[idx], y_grid[idx] = _regularize_zxy(e, x, y, projected)
    return elevation, x_grid, y_grid

#%% # Function to show elevation in a isometric plot
def plot_elevation_isometric(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
        figure: plt.figure=None,
        show: bool=True
    ) -> tuple[plt.figure, plt.axes]:
    """
    Show elevation in a isometric plot.
    
    Args:
        elevation (list | pd.Series | np.ndarray): The elevation data, which can be a single matrix or a list of matrices.
        x_grid (list | pd.Series | np.ndarray): The x coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        y_grid (list | pd.Series | np.ndarray): The y coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        figure (plt.figure): The figure to plot on.
        show (bool): Whether to show the plot.
        
    Returns:
        tuple[plt.figure, plt.axes]: The figure and axes.
    """
    elevation, x_grid, y_grid = _regularize_zxy_multi(elevation, x_grid, y_grid)

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

#%% # Function to show elevation in a 2D plot
def plot_elevation_2d(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
        figure: plt.figure=None,
        show: bool=True
    ) -> tuple[plt.figure, plt.axes]:
    """
    Show elevation in a 2D plot.

    Args:
        elevation (list | pd.Series | np.ndarray): The elevation data, which can be a single matrix or a list of matrices.
        x_grid (list | pd.Series | np.ndarray): The x coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        y_grid (list | pd.Series | np.ndarray): The y coordinates of the grid, which must be provided when elevation is split into multiple matrices.
        figure (plt.figure): The figure to plot on.
        show (bool): Whether to show the plot.
        
    Returns:
        tuple[plt.figure, plt.axes]: The figure and axes.
    """
    elevation, x_grid, y_grid = _regularize_zxy_multi(elevation, x_grid, y_grid)

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

#%% # Function to show elevation in a 3D interactive plot
def plot_elevation_3d(
        elevation: list | pd.Series | np.ndarray, 
        x_grid: list | pd.Series | np.ndarray=None, 
        y_grid: list | pd.Series | np.ndarray=None,
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
        projected (bool, optional): If True, the x_grid and y_grid will be converted to projected coordinates.
        figure (mlab.figure, optional): The figure to plot on.
        show (bool, optional): Whether to show the plot.

    Returns:
        tuple[mlab.figure, mlab.axes]: The figure and axes.
    """
    elevation, x_grid, y_grid = _regularize_zxy_multi(elevation, x_grid, y_grid, projected)
    
    # create a unique color bar for all dtms
    min_elv = min([e.min() for e in elevation])
    max_elv = max([e.max() for e in elevation])

    if figure is None:
        fig = mlab.figure()
    else:
        fig = figure
    axs = mlab.axes(figure=fig)
    
    for e, x, y in zip(elevation, x_grid, y_grid):
        mlab.mesh(x, y, e, colormap='terrain', vmin=min_elv, vmax=max_elv, figure=fig)

    if show: mlab.show()
    return fig, axs
    
#%%
