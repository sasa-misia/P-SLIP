#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prtoject configuration file.
This file contains the configuration settings for the project, like
folder structure, default parameters, and other settings.
"""

# %% === Default parameters and configurations
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
    'rain',
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

# %% === Default table for standardized and parameterized classes
CLASS_NUMBER_DIGITS = 2

# Main properties association with standardized topsoil, subsoil, vegetation, and land use
# This table is useful in case of ML models
STANDARD_CLASSES_FILENAME = 'standard_classes.csv'
STANDARD_CLASS_NAME_MAPPER = {
    'soil_sub': [
        'ssl', 
        {
            0:  ['Unknown', 'Class not defined or not classified.'],
            1:  ['Cohesive soil', 'Fine-grained soils, mainly clays, with cohesive properties.'],
            2:  ['Soft non-cohesive soil', 'Loose silts and sands, low cohesion, easily erodible.'],
            3:  ['Non-cohesive soil', 'Gravel, coarse-grained, high permeability, no cohesion.'],
            4:  ['Rock', 'Unconsolidated or fractured rock, moderate strength.'],
            5:  ['Strong rock', 'Massive, consolidated rock, high strength and stability.']
        }
    ],
    'soil_top': [
        'tsl',
        {
            0:  ['Unknown', 'Class not defined or not classified.'],
            1:  ['Clay', 'Fine-grained topsoil, high plasticity, low permeability.'],
            2:  ['Silty clay', 'Mix of silt and clay, intermediate properties.'],
            3:  ['Clay loam', 'Balanced clay and loam, moderate fertility.'],
            4:  ['Loam', 'Ideal agricultural soil, balanced sand, silt, clay.'],
            5:  ['Silty clay loam', 'Loam with higher silt and clay content.'],
            6:  ['Silt loam', 'Loam with higher silt content, good drainage.'],
            7:  ['Silt', 'Fine particles, high erosion risk, moderate fertility.'],
            8:  ['Sandy clay', 'Clay with significant sand, improved drainage.'],
            9:  ['Sandy clay loam', 'Loam with sand and clay, good structure.'],
            10: ['Loamy sand', 'Mostly sand, some loam, high drainage.'],
            11: ['Sandy loam', 'Loam with high sand content, well-drained.'],
            12: ['Sand', 'Coarse particles, very high drainage, low fertility.']
        }
    ],
    'vegetation': [
        'veg',
        {
            0:  ['Unknown', 'Class not defined or not classified.'],
            1:  ['Grass', 'Low vegetation, ground cover, erosion control.'],
            2:  ['Bush', 'Shrubs and bushes, medium height, moderate root system.'],
            3:  ['Low tree', 'Trees 3-10 m, small canopy, shallow roots.'],
            4:  ['Medium tree', 'Trees 10-20 m, moderate canopy, deeper roots.'],
            5:  ['High tree', 'Trees 20-30 m, large canopy, extensive root system.'],
            6:  ['Very high and robust tree', 'Trees >30 m, very large canopy, strong root system.']
        }
    ],
    'land_use': [
        'lnd',
        {
            0:  ['Unknown', 'Class not defined or not classified.'],
            1:  ['Bare soil', 'Exposed soil, no vegetation, high erosion risk.'],
            2:  ['Sloping man-made areas', 'Artificial slopes, embankments, infrastructure.'],
            3:  ['Mountainous and rocky areas', 'Natural rocky terrain, mountains, high relief.'],
            4:  ['Various crops', 'Agricultural areas with mixed crops.'],
            5:  ['Low vegetation and meadows', 'Natural meadows, grasslands, low vegetation.'],
            6:  ['Woods (Forests)', 'Forested areas, dense tree cover.'],
            7:  ['Flat man-made areas', 'Urban or industrial flat surfaces, paved or built.'],
            8:  ['Submerged areas', 'Areas underwater, lakes, reservoirs, wetlands.']
        }
    ]
}
# Full, structured mapping of standardized classes
DEFAULT_STANDARD_CLASSES = [
    {
        'type': type_key,
        'class_name': cls_dict[0],
        'class_id': f"{STANDARD_CLASS_NAME_MAPPER[type_key][0]}{str(cls_id).zfill(CLASS_NUMBER_DIGITS)}",
        'class_num': cls_id,
        'info': cls_dict[1]
    }
    for type_key in STANDARD_CLASS_NAME_MAPPER
    for cls_id, cls_dict in STANDARD_CLASS_NAME_MAPPER[type_key][1].items()
]

# Main properties association with parameter classes
# This table is useful in case of physical modeling
PARAMETER_CLASSES_FILENAME = 'parameter_classes.csv'
PARAMETER_CLASS_NAME_MAPPER = {
    'units':[
        'uom',
        {
            0: {'GS':'-', 'gd':'kN/m3', 'c':'kPa', 'cr':'kPa', 'phi':'deg', 'kt':'m/h', 'beta':'-', 'A':'-',  'lambda':'-', 'n':'-',  'E':'MPa',  'ni':'-',  'info':'Unit of measurement'},
        }
    ],
    'soil': [
        'slp',
        {
            1: {'GS':2.65, 'gd':16.5, 'c':0,   'cr':0, 'phi':32, 'kt':1e-2, 'beta':0, 'A':40,  'lambda':0.4, 'n':0.4,  'E':20,  'ni':0.30, 'info':'Loose sandy soils - High permeability, no cohesion'},
            2: {'GS':2.67, 'gd':18.0, 'c':0,   'cr':0, 'phi':38, 'kt':5e-3, 'beta':0, 'A':40,  'lambda':0.4, 'n':0.35, 'E':60,  'ni':0.25, 'info':'Dense sandy soils - Medium permeability, no cohesion'},
            3: {'GS':2.65, 'gd':17.5, 'c':5,   'cr':0, 'phi':18, 'kt':1e-3, 'beta':0, 'A':80,  'lambda':0.4, 'n':0.45, 'E':20,  'ni':0.35, 'info':'Loamy soils - Moderate cohesion, mixed grain size'},
            4: {'GS':2.68, 'gd':18.0, 'c':10,  'cr':0, 'phi':22, 'kt':5e-4, 'beta':0, 'A':80,  'lambda':0.4, 'n':0.48, 'E':15,  'ni':0.30, 'info':'Silty soils - Low permeability, moderate cohesion'},
            5: {'GS':2.70, 'gd':15.0, 'c':15,  'cr':0, 'phi':20, 'kt':1e-4, 'beta':0, 'A':100, 'lambda':0.4, 'n':0.52, 'E':10,  'ni':0.35, 'info':'Normally-consolidated clayey soils - High plasticity and soft'},
            6: {'GS':2.72, 'gd':19.0, 'c':25,  'cr':0, 'phi':25, 'kt':5e-5, 'beta':0, 'A':100, 'lambda':0.4, 'n':0.45, 'E':30,  'ni':0.30, 'info':'Over-consolidated clayey soils - Preloaded, higher strength and stiffer'},
            7: {'GS':2.65, 'gd':25.0, 'c':3e4, 'cr':0, 'phi':35, 'kt':1e-3, 'beta':0, 'A':0,   'lambda':0.4, 'n':0.25, 'E':1e4, 'ni':0.20, 'info':'Rocky soils - Fractured rock with soil matrix or compacted rock'},
            8: {'GS':2.60, 'gd':18.0, 'c':0,   'cr':0, 'phi':40, 'kt':2e-2, 'beta':0, 'A':0,   'lambda':0.4, 'n':0.3,  'E':120, 'ni':0.27, 'info':'Gravel soils - Coarse-grained, high permeability'}
        }
    ],
    'vegetation': [
        'vgp',
        {
            1: {'GS':0, 'gd':0, 'c':0,  'cr':2,  'phi':0, 'kt':0, 'beta':0.5, 'A':0, 'lambda':0, 'n':0, 'E':0, 'ni':0, 'info':'Grass - Shallow roots, low cohesion contribution'},
            2: {'GS':0, 'gd':0, 'c':0,  'cr':5,  'phi':0, 'kt':0, 'beta':0.6, 'A':0, 'lambda':0, 'n':0, 'E':0, 'ni':0, 'info':'Bushes - Moderate root cohesion, medium depth'},
            3: {'GS':0, 'gd':0, 'c':0,  'cr':10, 'phi':0, 'kt':0, 'beta':0.6, 'A':0, 'lambda':0, 'n':0, 'E':0, 'ni':0, 'info':'Beech trees - Deciduous trees, moderate root strength'},
            4: {'GS':0, 'gd':0, 'c':0,  'cr':15, 'phi':0, 'kt':0, 'beta':0.6, 'A':0, 'lambda':0, 'n':0, 'E':0, 'ni':0, 'info':'Firs - Coniferous trees, strong root system'},
            5: {'GS':0, 'gd':0, 'c':0,  'cr':20, 'phi':0, 'kt':0, 'beta':0.6, 'A':0, 'lambda':0, 'n':0, 'E':0, 'ni':0, 'info':'Oaks - Deep rooting, high cohesion contribution'}
        }
    ]
}
# Full, structured mapping of parameterized classes
DEFAULT_PARAMETER_CLASSES = [
    {
        'type': type_key,
        'class_id': f"{PARAMETER_CLASS_NAME_MAPPER[type_key][0]}{str(cls_id).zfill(CLASS_NUMBER_DIGITS)}",
        'class_num': cls_id,
        **params,
    }
    for type_key in PARAMETER_CLASS_NAME_MAPPER
    for cls_id, params in PARAMETER_CLASS_NAME_MAPPER[type_key][1].items()
]

# %% === Default reference points csv
REFERENCE_POINTS_FILENAME = 'reference_points.csv'
REFERENCE_POINTS_CVS_COLUMNS = ['lon', 'lat', 'id', 'date', 'info']

# %% === dictionary of supported file types
SUPPORTED_FILE_TYPES = {
    'vectorial': ['.shp', '.gpkg', '.geojson', '.sqlite'], 
    'table': ['.csv'],
    'raster': ['.tif', '.tiff'],
    'climate': ['.nc']
}