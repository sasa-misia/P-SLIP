#%% # Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import warnings
import mayavi.mlab as mlab

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
        y_grid: np.ndarray=None
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
    _check_grid_consistency(elevation, x_grid, y_grid)
    return elevation, x_grid, y_grid

#%% # Function to show elevation in a isometric plot
def show_elevation_isometric(
        elevation: np.ndarray, 
        x_grid: np.ndarray=None, 
        y_grid: np.ndarray=None
    ) -> None:
    """
    Show elevation in a isometric plot.
    
    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        
    Returns:
        None
    """
    elevation, x_grid, y_grid = _regularize_zxy(elevation, x_grid, y_grid)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, elevation, cmap='viridis')
    plt.show()

#%% # Function to show elevation in a 2D plot
def show_elevation_2d(
        elevation: np.ndarray, 
        x_grid: np.ndarray=None, 
        y_grid: np.ndarray=None
    ) -> None:
    """
    Show elevation in a 2D plot.

    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.
        
    Returns:
        None
    """
    elevation, x_grid, y_grid = _regularize_zxy(elevation, x_grid, y_grid)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Elevation Map')
    ax.imshow(elevation, cmap='viridis', extent=(x_grid[0, 0], x_grid[-1, -1], y_grid[0, 0], y_grid[-1, -1]))
    ax.set_aspect('equal')
    plt.show()

#%% # Function to show elevation in a 3D interactive plot
def show_elevation_3d(
        elevation: np.ndarray, 
        x_grid: np.ndarray=None, 
        y_grid: np.ndarray=None
    ) -> None:
    """
    Show elevation in a 3D interactive plot.

    Args:
        elevation (np.ndarray): The elevation data.
        x_grid (np.ndarray): The x coordinates of the grid.
        y_grid (np.ndarray): The y coordinates of the grid.

    Returns:
        None
    """
    elevation, x_grid, y_grid = _regularize_zxy(elevation, x_grid, y_grid)
    mlab.mesh(x_grid, y_grid, elevation, colormap='terrain')
    mlab.show()
    
# %%
