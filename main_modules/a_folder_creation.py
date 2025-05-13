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
from typing import Dict, Optional

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
    os_slash: str
    creation_time: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Main directory
    base_dir: str = field(default_factory=os.getcwd)
    
    # Folder paths
    inp_dir: Optional[str] = None
    var_dir: Optional[str] = None
    res_dir: Optional[str] = None
    usr_dir: Optional[str] = None
    out_dir: Optional[str] = None
    log_dir: Optional[str] = None
    
    # Subfolder paths
    inp_sub: Dict[str, str] = field(default_factory=dict)
    out_sub: Dict[str, str] = field(default_factory=dict)
    res_sub: Dict[str, str] = field(default_factory=dict)
    
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


def check_libraries(required_file=config.LIBRARIES_CONFIG['required_file'], optional_file=config.LIBRARIES_CONFIG['optional_file']):
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
    
    with open(config_dir / required_file, 'r') as f:
        required_libs = [line.strip() for line in f.readlines() if line.strip()] # Remove empty lines and strip whitespace
    
    with open(config_dir / optional_file, 'r') as f:
        optional_libs = [line.strip() for line in f.readlines() if line.strip()] # Remove empty lines and strip whitespace
    
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
    check_libraries()
    logger.info("Check of libraries completed.")
    
    # If no case name is provided, use the default
    if case_name is None:
        case_name = config.DEFAULT_CASE_NAME
    
    # Create the analysis environment object
    env = AnalysisEnvironment(
        case_name=case_name,
        user=user,
        os_slash=sl
    )
    
    # Select the base directory
    # If no base directory is provided, ask the user for it
    if base_dir is None:
        base_dir = input("Enter the base directory for the analysis (or press Enter to use the current directory): ")
        if not base_dir.strip(): # If the user pressed Enter and leave it blank, use the current directory
            base_dir = os.getcwd()
    elif not os.path.isdir(base_dir):
        raise ValueError(f"The specified base directory does not exist: {base_dir}")
    
    env.base_dir = base_dir
    
    # Main directories
    env.inp_dir = os.path.join(base_dir, 'inputs')
    env.var_dir = os.path.join(base_dir, 'variables')
    env.res_dir = os.path.join(base_dir, 'results')
    env.usr_dir = os.path.join(base_dir, 'user_control')
    env.out_dir = os.path.join(base_dir, 'outputs')
    env.log_dir = os.path.join(base_dir, 'logs')
    
    # Create the main directories if they do not exist
    main_dirs = [env.inp_dir, env.var_dir, env.res_dir, 
                    env.usr_dir, env.out_dir, env.log_dir]
    
    for dir_path in main_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"New main folder created: {dir_path}")
    
    # Create the input subdirectories if they do not exist
    inp_sub = config.ANALYSIS_FOLDER_STRUCTURE['inputs']
    
    for subfolder in inp_sub:
        path = os.path.join(env.inp_dir, subfolder)
        env.inp_sub[subfolder] = path
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"New input folder created: {path}")
    
    # Create the output subdirectories if they do not exist
    out_sub = config.ANALYSIS_FOLDER_STRUCTURE['outputs']
    
    for subfolder in out_sub:
        path = os.path.join(env.out_dir, subfolder)
        env.out_sub[subfolder] = path
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"New output folder created: {path}")
    
    # Create the result subdirectories if they do not exist
    res_sub = config.ANALYSIS_FOLDER_STRUCTURE['results']
    
    for subfolder in res_sub:
        path = os.path.join(env.res_dir, subfolder)
        env.res_sub[subfolder] = path
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"New result folder created: {path}")
    
    # Create the input_files.csv file
    # This file will be used to store the paths of the input files
    input_files_df = pd.DataFrame(columns=config.INPUT_FILES_COLUMNS)
    input_files_path = os.path.join(env.inp_dir, 'input_files.csv')
    input_files_df.to_csv(input_files_path, index=False)
    logger.info(f"File input_files.csv created: {input_files_path}")
    
    # Save the environment to a JSON file
    # This file will contain the details of the analysis environment
    # and can be used for future reference or loading
    env_file_path = os.path.join(env.base_dir, 'analysis_environment.json')
    env.to_json(env_file_path)
    logger.info(f"Analysis environment saved to: {env_file_path}")
    
    logger.info(f"Analysis folder structure created successfully for: {case_name}")
    return env


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


if __name__ == "__main__":
    # Parse command line arguments
    # This allows the script to be run from the command line with parameters
    parser = argparse.ArgumentParser(description="Create the folder structure for the analysis.")
    parser.add_argument("--case_name", help="Case study name", default=None)
    parser.add_argument("--base_dir", help="Base directory for the analysis", default=None)
    args = parser.parse_args()
    
    # Call the main function with the provided arguments
    main(args.case_name, base_dir=args.base_dir)