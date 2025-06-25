#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prtoject configuration file.
This file contains the configuration settings for the project, like
folder structure, default parameters, and other settings.
"""

# Default case name
# This is used when no case name is provided by the user.
DEFAULT_CASE_NAME = 'Not Defined - Standalone'

# Environment filename
# This is the name of the file that contains the analysis environment details.
ENVIRONMENT_FILENAME = 'analysis_environment.json'

# Configuration for the folder structure
# This dictionary defines the folder structure for the project.
ANALYSIS_FOLDER_STRUCTURE = {
    'inputs': [
        {'rainfalls': [
            'recordings', 
            'forecast'
        ]},
        'landslides',
        'study_area',
        'satellite_images',
        'dtm',
        'soil',
        'vegetation',
        'roads',
        'land_uses',
        {'temperature': [
            'recordings', 
            'forecast'
        ]},
        'miscellaneous'
    ],
    'variables': [],
    'results': [
        'safety_factors',
        'machine_learning',
        'evolution'
    ],
    'user_control': [],
    'outputs': [
        {'figures': [
            'susceptibility_maps',
        ]},
        'tables'
    ],
    'logs': []
}

# Dynamically create main directories and nested subfolders from ANALYSIS_FOLDER_STRUCTURE 
# Remember to modify in case of changes in ANALYSIS_FOLDER_STRUCTURE!
ANALYSIS_FOLDER_ATTRIBUTE_MAPPER = {
    'inputs': 'inp_dir',
    'variables': 'var_dir',
    'results': 'res_dir',
    'user_control': 'usr_dir',
    'outputs': 'out_dir',
    'logs': 'log_dir'
}

# --- CHECK: ANALYSIS_FOLDER_ATTRIBUTE_MAPPER vs ANALYSIS_FOLDER_STRUCTURE ---
# Just the first level keys of ANALYSIS_FOLDER_STRUCTURE should be in ANALYSIS_FOLDER_ATTRIBUTE_MAPPER.
# This check ensures that the keys in ANALYSIS_FOLDER_ATTRIBUTE_MAPPER match the top-level keys
structure_keys = set(ANALYSIS_FOLDER_STRUCTURE.keys())
map_keys = set(ANALYSIS_FOLDER_ATTRIBUTE_MAPPER.keys())
if structure_keys != map_keys:
    raise ValueError(
        f"Mismatch between ANALYSIS_FOLDER_ATTRIBUTE_MAPPER keys and ANALYSIS_FOLDER_STRUCTURE keys:\n"
        f"  Only in ANALYSIS_FOLDER_ATTRIBUTE_MAPPER: {map_keys - structure_keys}\n"
        f"  Only in ANALYSIS_FOLDER_STRUCTURE: {structure_keys - map_keys}"
    )
# --- END CHECK ---

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

# Raw input files configuration
# This is the name of the CSV file that contains the raw input files.
RAW_INPUT_FILENAME = 'input_files.csv'
# This dictionary contains the expected columns for various input files.
RAW_INPUT_CSV_COLUMNS = ['path', 'type', 'internal']

# Libraries to check
# This dictionary contains the required and optional libraries for the project.
LIBRARIES_CONFIG = {
    'required_file': 'requirements.txt',
    'optional_file': 'requirements_opt.txt'
}