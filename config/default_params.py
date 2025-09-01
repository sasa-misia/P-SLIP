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

# Input types and subfolders for dynamic inputs
KNOWN_REQUIRED_STATIC_INPUT_TYPES = [
    'study_area',
    'dtm'
]
KNOWN_OPTIONAL_STATIC_INPUT_TYPES = [
    'landslides',
    'soil',
    'vegetation',
    'infrastructures',
    'land_use'
]
KNOWN_DYNAMIC_INPUT_TYPES = [
    'rainfalls', 
    'temperature'
]
DYNAMIC_SUBFOLDERS = [
    'recordings', 
    'forecast'
]
GENERIC_INPUT_TYPE = ['miscellaneous']

# Configuration for the folder structure
# This dictionary defines the folder structure for the project.
ANALYSIS_FOLDER_STRUCTURE = {
    'inputs': [
        *KNOWN_REQUIRED_STATIC_INPUT_TYPES,
        *KNOWN_OPTIONAL_STATIC_INPUT_TYPES,
        {name: DYNAMIC_SUBFOLDERS for name in KNOWN_DYNAMIC_INPUT_TYPES},
        *GENERIC_INPUT_TYPE
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

# Raw input files configuration
# This is the name of the CSV file that contains the raw input files.
RAW_INPUT_FILENAME = 'input_files.csv'
# This dictionary contains the expected columns for various input files.
RAW_INPUT_CSV_COLUMNS = ['custom_id', 'path', 'type', 'subtype', 'internal']

# Libraries to check
# This dictionary contains the required and optional libraries for the project.
LIBRARIES_CONFIG = {
    'required_file': 'requirements.txt',
    'optional_file': 'requirements_opt.txt'
}

# Plotting configuration
# This dictionary contains default settings for plotting.
DEFAULT_PLOT_CONFIG = {
    'default': {
        'figsize': (10, 6),
        'dpi': 300,
        'style': 'seaborn-v0_8-whitegrid',
        'cmap': 'viridis'
    }
}

# Logging configuration
# This dictionary contains default settings for logging.
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
}

# User analysis configuration
# This dictionary contains the analysis configuration template.
ANALYSIS_CONFIGURATION = {
    'inputs': { # Here all the settings used to import the various files (ex: {'study_area': [{'settings': {}, 'custom_id': []}]})
        **{k: [{'settings': {}, 'custom_id': []}] for k in KNOWN_REQUIRED_STATIC_INPUT_TYPES},
        **{k: [{'settings': {}, 'custom_id': []}] for k in KNOWN_OPTIONAL_STATIC_INPUT_TYPES},
        **{k: [{'settings': {}, 'custom_id': []}] for k in KNOWN_DYNAMIC_INPUT_TYPES},
        **{k: [{'settings': {}, 'custom_id': []}] for k in GENERIC_INPUT_TYPE}
    },
    'variables': { # Here a list of variables and content (ex: {'variable.pkl': {'var1', 'var2', 'var3'}}})
        'example.pkl': [
            'example_var1',
            'example_var2', 
            'example_var3'
        ]
    },
    'results': { # Here all the settings for the results are defined
        'example_mdl_name': {
            'settings': {},
            'folder': None 
        }
    },
    'outputs': {
        'figures': { # Here all the settings for the figures are defined
            'settings': DEFAULT_PLOT_CONFIG
        },
        'tables': { # Here all the settings for the tables are defined
            'settings': {}
        }
    },
    'logs': { # Here all the settings for the logs are defined
        'settings': LOG_CONFIG
    }
}