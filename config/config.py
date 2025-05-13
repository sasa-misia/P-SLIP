#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prtoject configuration file.
This file contains the configuration settings for the project, like
folder structure, default parameters, and other settings.
"""

# Default case name
# This is used when no case name is provided by the user.
DEFAULT_CASE_NAME = 'ND - Standalone'

# Configuration for the folder structure
# This dictionary defines the folder structure for the project.
ANALYSIS_FOLDER_STRUCTURE = {
    'inputs': [
        'rainfalls',
        'rain_forecast',
        'detected_soil_slips',
        'municipalities',
        'lithology',
        'satellite_images',
        'roads',
        'land_uses',
        'dtm',
        'vegetation',
        'temperature',
        'temp_forecast'
    ],
    'variables': [],
    'results': [
        'factors_of_safety',
        'ml_models_and_predictions',
        'flow_paths'
    ],
    'user_control': [],
    'outputs': [
        'figures',
        'tables'
    ],
    'logs': []
}

# Default parameters for modules
# This dictionary contains default parameters for various modules.
DEFAULT_PARAMS = {
    'folder_creation': {
        'case_name': DEFAULT_CASE_NAME
    },
    # Add here other modules and their default parameters
}

# Plotting configuration
# This dictionary contains default settings for plotting.
PLOT_CONFIG = {
    'default_figsize': (10, 6),
    'default_dpi': 300,
    'default_style': 'seaborn-v0_8-whitegrid',
    'default_cmap': 'viridis'
}

# Logging configuration
# This dictionary contains default settings for logging.
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Input files configuration
# This dictionary contains the expected columns for various input files.
INPUT_FILES_COLUMNS = ['path', 'type', 'internal']

# Libraries to check
# This dictionary contains the required and optional libraries for the project.
LIBRARIES_CONFIG = {
    'required_file': 'required_libraries.txt',
    'optional_file': 'optional_libraries.txt'
}