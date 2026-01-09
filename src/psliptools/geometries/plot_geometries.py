# %% === Setting matplotlib to plot on external windows
import matplotlib
matplotlib.use('Qt5Agg') # To plot in external window and not in interactive terminal

# %% === Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shapely
import warnings
from typing import List, Tuple, Union, Optional
import pandas as pd

# %% === Function to plot a list of polygons
def plot_polygons(
        polygons: Union[List[shapely.geometry.base.BaseGeometry], pd.Series],
        colors: Optional[List[str]] = None,
        figure: Optional[plt.Figure] = None,
        show: bool = True,
        alpha: float = 0.7
    ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a list of polygons with random or custom colors.
    
    Args:
        polygons (Union[List[shapely.geometry.base.BaseGeometry], pd.Series]): List of Shapely polygons or pandas Series to plot.
        colors (Optional[List[str]]): List of colors for each polygon. If None, random colors are generated.
        figure (Optional[plt.Figure]): The figure to plot on.
        show (bool): Whether to show the plot.
        alpha (float): Transparency level for the polygons.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes.
    """
    # Convert pandas Series to list
    if isinstance(polygons, pd.Series):
        polygons = polygons.to_list()
    
    if not isinstance(polygons, list):
        raise ValueError(f"polygons must be a list or pandas Series, not {type(polygons)}")
    
    if colors is not None and len(colors) != len(polygons):
        raise ValueError(f"colors must have the same length as polygons ({len(polygons)}), not {len(colors)}")
    
    if figure is None:
        fig = plt.figure()
    else:
        fig = figure
    
    axs = fig.add_subplot(111)
    axs.set_title('Polygons Map')
    axs.set_aspect('equal')
    
    # Get available matplotlib colors if needed for random selection
    available_colors = list(mcolors.CSS4_COLORS.keys())
    
    for i, polygon in enumerate(polygons):
        if polygon is None:
            warnings.warn(f"Polygon at index {i} is None, skipping...", stacklevel=2)
            continue
            
        if colors:
            color = colors[i]
        else:
            # Generate random color from available matplotlib colors
            color = available_colors[np.random.randint(0, len(available_colors))]
        
        if isinstance(polygon, shapely.geometry.MultiPolygon):
            for poly in polygon.geoms:
                x, y = poly.exterior.xy
                axs.fill(x, y, color=color, alpha=alpha)
                for interior in poly.interiors:
                    x_int, y_int = interior.xy
                    axs.fill(x_int, y_int, color='white', alpha=1.0)
        elif isinstance(polygon, shapely.geometry.Polygon):
            x, y = polygon.exterior.xy
            axs.fill(x, y, color=color, alpha=alpha)
            for interior in polygon.interiors:
                x_int, y_int = interior.xy
                axs.fill(x_int, y_int, color='white', alpha=1.0)
        else:
            warnings.warn(f"Polygon at index {i} is not a valid Shapely Polygon or MultiPolygon, skipping...", stacklevel=2)
    
    if show:
        plt.show()
    
    return fig, axs

# %%
