#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for creating the folder structure for the analysis.
"""

import os
import sys
import platform
import importlib
import warnings
import json
from pathlib import Path
import logging
import argparse
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any

# Add the parent directory to the system path (temporarily)
# This allows importing modules from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import default configuration
from config import config

@dataclass
class AnalysisEnvironment:
    """Class to store the analysis environment details."""
    
    # Metadata
    case_name: str
    user: str
    os_separator: str
    creation_time: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Main directory as dict with 'path'
    base_dir: Dict[str, str] = field(default_factory=lambda: {'path': os.getcwd()})
    
    # Folder paths as nested dicts
    # Remember to modify in case of changes in ANALYSIS_FOLDER_STRUCTURE!
    inp_dir: Dict[str, Any] = field(default_factory=dict)
    var_dir: Dict[str, Any] = field(default_factory=dict)
    res_dir: Dict[str, Any] = field(default_factory=dict)
    usr_dir: Dict[str, Any] = field(default_factory=dict)
    out_dir: Dict[str, Any] = field(default_factory=dict)
    log_dir: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self, file_path):
        """Save the environment to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def from_json(cls, file_path):
        """Load the environment from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# The following method checks if the required and optional libraries are installed (private).
def _check_libraries(required_file=config.LIBRARIES_CONFIG['required_file'], optional_file=config.LIBRARIES_CONFIG['optional_file']):
    """
    Verify if the required and optional libraries are installed.
    If any required library is missing, an ImportError is raised.
    
    Args:
        required_file: Path to the TXT file with required libraries
        optional_file: Path to the TXT file with optional libraries
    
    Returns:
        tuple: (required_status, optional_status) - True if all libraries are installed, False otherwise
    """
    # Load the configuration files
    # Assuming the config files are in the parent directory of this script
    config_dir = Path(__file__).parent.parent

    def _parse_libs(file_path):
        with open(file_path, 'r') as f:
            return [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.strip().startswith('#')
            ]

    required_libs = _parse_libs(config_dir / required_file)
    optional_libs = _parse_libs(config_dir / optional_file)
    
    # Check required libraries
    # Try to import each library and catch ImportError
    missing_required = []
    for lib in required_libs:
        try:
            importlib.import_module(lib.split('.')[0])  # Try to import the main module (e.g., 'numpy' instead of 'numpy.linalg')
        except ImportError:
            missing_required.append(lib)
    
    # Check optional libraries
    # Try to import each library and catch ImportError
    missing_optional = []
    for lib in optional_libs:
        try:
            importlib.import_module(lib.split('.')[0])  # Try to import the main module (e.g., 'numpy' instead of 'numpy.linalg')
        except ImportError:
            missing_optional.append(lib)
    
    # Show error if required libraries are missing
    if missing_required:
        error_msg = "It is necessary to install the following libraries:\n" + "\n".join(missing_required)
        raise ImportError(error_msg)
    
    # Show warning if optional libraries are missing
    if missing_optional:
        warnings.warn("It is suggested to install the following libraries:\n" + "\n".join(missing_optional))
    
    return (len(missing_required) == 0, len(missing_optional) == 0)


# The following method creates nested folders based on a given structure (private).
def _create_nested_folders(base_path, structure, logger):
    """
    Recursive helper to create nested folders and return a dict with their paths.

    Args:
        base_path: the parent directory
        structure: list of str or dict (for subfolders)

    Returns: dict with 'path' and subfolder keys
    """
    result = {'path': base_path}
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        logger.info(f"New folder created: {base_path}")
    if isinstance(structure, list):
        for item in structure:
            if isinstance(item, str):
                sub_path = os.path.join(base_path, item)
                result[item] = {'path': sub_path}
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                    logger.info(f"New subfolder created: {sub_path}")
            elif isinstance(item, dict):
                for subkey, subval in item.items():
                    sub_path = os.path.join(base_path, subkey)
                    result[subkey] = _create_nested_folders(sub_path, subval, logger)
    return result


# The following method creates the folder structure for the analysis (public).
def create_folder_structure(case_name=None, base_dir=None):
    """
    Create the folder structure for the analysis.
    This function creates the main analysis directory and subdirectories
    
    Args:
        case_name: Name of the case study. If None, a default name is used.
        base_dir: Base directory for the analysis. If None, the current directory is used.
    
    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format=config.LOG_CONFIG['format'], 
                        datefmt=config.LOG_CONFIG['date_format'])
    logger = logging.getLogger(__name__)
    
    # OS detection
    if platform.system() == 'Windows':
        user = os.environ.get('USERNAME', 'unknown') # If USERNAME is not set, use 'unknown'
    else:  # Linux, macOS, etc.
        user = os.environ.get('USER', 'unknown') # If USERNAME is not set, use 'unknown'
        if platform.system() == 'Darwin':  # macOS
            pass  # Nothing specific to do here
        else:  # Linux or others
            logger.warning('Platform not yet tested, In case of problems, please contact the developer.')
    
    # Separator
    sl = os.path.sep

    # Library check
    logger.info("Check of required and optional libraries...")
    _check_libraries()
    logger.info("Check of libraries completed.")

    # If no case name is provided, use the default
    if case_name is None:
        case_name = config.DEFAULT_CASE_NAME

    # Create the analysis environment object
    env = AnalysisEnvironment(
        case_name=case_name,
        user=user,
        os_separator=sl
    )

    # Select the base directory
    if base_dir is None:
        base_dir = input("Enter the base directory for the analysis (or press Enter to use the current directory): ")
        if not base_dir.strip():
            base_dir = os.getcwd()
    elif not os.path.isdir(base_dir):
        raise ValueError(f"The specified base directory does not exist: {base_dir}")

    env.base_dir = {'path': base_dir}

    # Dynamically create main directories and nested subfolders from ANALYSIS_FOLDER_STRUCTURE 
    # Remember to modify in case of changes in ANALYSIS_FOLDER_STRUCTURE!
    folder_attr_map = {
        'inputs': 'inp_dir',
        'variables': 'var_dir',
        'results': 'res_dir',
        'user_control': 'usr_dir',
        'outputs': 'out_dir',
        'logs': 'log_dir'
    }
    for key, attr in folder_attr_map.items():
        main_path = os.path.join(base_dir, key)
        structure = config.ANALYSIS_FOLDER_STRUCTURE.get(key, [])
        nested_dict = _create_nested_folders(main_path, structure, logger)
        setattr(env, attr, nested_dict)

    # Create the input_files.csv file in the main input folder
    input_files_df = pd.DataFrame(columns=config.INPUT_FILES_COLUMNS)
    input_files_path = env.inp_dir['path'] if 'path' in env.inp_dir else os.path.join(base_dir, 'inputs')
    input_files_path = os.path.join(input_files_path, 'input_files.csv')
    input_files_df.to_csv(input_files_path, index=False)
    logger.info(f"File input_files.csv created: {input_files_path}")
    
    # Save the environment to a JSON file
    # This file will contain the details of the analysis environment
    # and can be used for future reference or loading
    env_file_path = os.path.join(env.base_dir['path'], 'analysis_environment.json')
    env.to_json(env_file_path)
    logger.info(f"Analysis environment saved to: {env_file_path}")
    
    logger.info(f"Analysis folder structure created successfully for: {case_name}")
    return env


# The following method is the main function of the module.
def main(case_name=None, gui_mode=False, base_dir=None):
    """
    Main function of the module.
    
    Args:
        case_name: Case study name. If None, a default name is used.
        gui_mode: If true, the function was called from a GUI.
        base_dir: Base directory for the analysis. If None, the current directory is used.
    
    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """
    if not gui_mode and case_name is None:
        case_name = input("Specify the analysis name (enter for [ND - Standalone]): ")
    
    env = create_folder_structure(case_name, base_dir)
    
    if not gui_mode:
        print(f"Analysis folder structure created successfully for: {case_name}")
    
    return env


# This block allows the script to be run from the command line with parameters.
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create the folder structure for the analysis.")
    parser.add_argument("--case_name", help="Case study name", default=None)
    parser.add_argument("--base_dir", help="Base directory for the analysis", default=None)
    args = parser.parse_args()
    
    # Call the main function with the provided arguments
    curr_env = main(case_name=args.case_name, base_dir=args.base_dir)