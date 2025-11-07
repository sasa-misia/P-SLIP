#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for creating the folder structure for the analysis.
"""

#%% Import necessary modules
import os
import re
import platform
import importlib
import json
import copy
import pickle
import shutil
import gzip
import bz2
import logging
import logging.handlers
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

try:
    import psutil
    SYSTEM_SPECS_AVAILABLE = True
except ImportError:
    SYSTEM_SPECS_AVAILABLE = False

GPU_SPECS_AVAILABLE = False # TODO: Implement libraries to obtain GPU specs

# Import default configuration
from .default_params import (
    RAW_INPUT_FILENAME,
    RAW_INPUT_CSV_COLUMNS,
    ENVIRONMENT_FILENAME,
    GENERIC_INPUT_TYPE,
    ANALYSIS_FOLDER_STRUCTURE,
    LIBRARIES_CONFIG,
    DEFAULT_CASE_NAME,
    ANALYSIS_CONFIGURATION,
    STANDARD_CLASSES_FILENAME,
    DEFAULT_STANDARD_CLASSES,
    PARAMETER_CLASSES_FILENAME,
    DEFAULT_PARAMETER_CLASSES,
    REFERENCE_POINTS_FILENAME,
    REFERENCE_POINTS_CVS_COLUMNS,
    LOG_CONFIG
)

from .version_writer import (
    get_app_version
)

# Import utility functions
from psliptools.utilities import parse_csv_internal_path_field, check_raw_path, add_row_to_csv

# %% === Logger ===
_log_memory_handler = logging.handlers.MemoryHandler(capacity=1000, flushLevel=logging.ERROR)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(LOG_CONFIG['format'], LOG_CONFIG['date_format']))

logging.basicConfig(
    level=logging.INFO,
    format=LOG_CONFIG['format'],
    datefmt=LOG_CONFIG['date_format'],
    handlers=[_log_memory_handler, _stream_handler]
)

def _setup_session_logger(logfile_path: str):
    """
    Set up the session logger, redirecting logs to both file and terminal.

    Args:
        logfile_path (str): Path to the log file.
    """
    logger = logging.getLogger()  # root logger, don't use getLogger(__name__)
    logger.info(f"Setting up session logger to file...")

    # Remove all FileHandler and StreamHandler from root logger
    for h in logger.handlers[:]:
        if isinstance(h, (logging.FileHandler, logging.StreamHandler)):
            logger.removeHandler(h)

    # Add new FileHandler
    file_handler = logging.FileHandler(logfile_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG['format'], LOG_CONFIG['date_format']))
    logger.addHandler(file_handler)

    # Add StreamHandler for terminal output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_CONFIG['format'], LOG_CONFIG['date_format']))
    logger.addHandler(stream_handler)

    # Move logs to file
    _log_memory_handler.setTarget(file_handler)
    _log_memory_handler.flush()

    logger.info(f"Logger redirected to: {logfile_path}")

# %% === Helper functions ===
def _retrieve_current_user() -> str:
    """
    Retrieve the current user.

    Returns:
        str: The current user.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Retrieving current user...")

    # OS detection
    if platform.system() == 'Windows':
        user = os.environ.get('USERNAME', 'unknown')
    else:
        user = os.environ.get('USER', 'unknown')
        logger.warning('Platform not yet tested. In case of problems, please contact the developer.')
    
    logger.info(f"Current user is: {user}")
    return user
        
def _retrieve_app_version() -> str:
    """
    Retrieve the application version.

    Returns:
        str: The application version.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Retrieving application version...")

    app_version = get_app_version()
    if app_version:
        logger.info(f"Application version is: {app_version}")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        version_file_path = os.path.join(script_dir, 'version.txt')
        with open(version_file_path, 'r') as f:
            app_version = f.read().strip()
            if app_version:
                logger.info(f"Application version ({app_version}) read from {version_file_path}")
            else:
                logger.warning(f"No application version found in {version_file_path}. Using default version 'unknown'.")
                app_version = 'unknown'
    return app_version

def _retrieve_system_specs() -> dict:
    """
    Retrieve system specs.

    Returns:
        dict: System specs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Retrieving system specs...")

    system_specs = {
        'python': platform.python_version(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0],
        'cores': None,
        'threads': None,
        'ram': None,
        'gpu_model': None,
        'gpu_vram': None
    }

    if SYSTEM_SPECS_AVAILABLE:
        system_specs['cores'] = psutil.cpu_count(logical=False)
        system_specs['threads'] = psutil.cpu_count(logical=True)
        system_specs['ram'] = psutil.virtual_memory().total / 1024**3
    
    if GPU_SPECS_AVAILABLE:
        # TODO: Implement GPU specs
        pass
    
    return system_specs

def _import_corrected_name(lib: str) -> str:
    """
    Import a library by its name, handling special cases for hyphenated names.
    """
    # Try to import the library using its name
    try:
        corrected_name = lib.split('.')[0].replace('-', '_')
        importlib.import_module(corrected_name)
        return corrected_name
    except ImportError:
        # If it fails, try manual mapping (add only special cases here)
        corrected_name = lib.split('.')[0]
        mapping = {
            "scikit-image": "skimage", # first the library name, then the import name
            "scikit-learn": "sklearn",
            "gdal": "osgeo"
        }
        if corrected_name in mapping:
            importlib.import_module(mapping[corrected_name])
            return mapping[corrected_name]
        # If it still fails, raise an error
        raise ImportError(
            f"Library '{lib}' could not be imported. Please ensure it is installed correctly or add entry to the mapping (eg: scikit-image: skimage)."
        )

def _check_libraries(required_file: str = LIBRARIES_CONFIG['required_file'],
                     optional_file: str = LIBRARIES_CONFIG['optional_file']) -> tuple[bool, bool]:
    """
    Verify if the required and optional libraries are installed.
    If any required library is missing, an ImportError is raised.

    Args:
        required_file (str): Path to the TXT file with required libraries.
        optional_file (str): Path to the TXT file with optional libraries.

    Returns:
        tuple: (required_status, optional_status) - True if all libraries are installed, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Checking required and optional libraries...")

    config_dir = Path(__file__).parent.parent

    def _parse_libs(file_path: Path) -> list[str]:
        with open(file_path, 'r') as f:
            return [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.strip().startswith('#')
            ]

    required_libs = _parse_libs(config_dir / required_file)
    optional_libs = _parse_libs(config_dir / optional_file)

    missing_required = []
    for lib in required_libs:
        try:
            _import_corrected_name(lib)
        except ImportError:
            missing_required.append(lib)

    missing_optional = []
    for lib in optional_libs:
        try:
            _import_corrected_name(lib)
        except ImportError:
            missing_optional.append(lib)

    if missing_required:
        error_msg = "The following required libraries must be installed: [" + "; ".join(missing_required) + "]"
        logger.error(error_msg)
        raise ImportError(error_msg)

    if missing_optional:
        warn_msg = "It is recommended to install the following optional libraries: [" + "; ".join(missing_optional) + "]"
        logger.warning(warn_msg)

    logger.info("Library check completed: required OK=%s, optional OK=%s", len(missing_required) == 0, len(missing_optional) == 0)
    return (len(missing_required) == 0, len(missing_optional) == 0)

def _create_nested_folders(base_path: str, structure: list) -> dict[str, object]:
    """
    Recursively create nested folders and return a dict with their paths.

    Args:
        base_path (str): The parent directory.
        structure (list): List of str or dict (for subfolders).

    Returns:
        dict: A dictionary with folder names as keys and their paths as values.
    """
    logger = logging.getLogger(__name__)
    result = {'path': base_path}
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        logger.info(f"New folder created: {base_path}")
    else:
        logger.info(f"Folder already exists: {base_path}")

    for item in structure:
        if isinstance(item, str):
            sub_path = os.path.join(base_path, item)
            result[item] = {'path': sub_path}
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
                logger.info(f"New subfolder created: {sub_path}")
            else:
                logger.info(f"Subfolder already exists: {sub_path}")
        elif isinstance(item, dict):
            for subkey, subval in item.items():
                sub_path = os.path.join(base_path, subkey)
                result[subkey] = _create_nested_folders(sub_path, subval)
    return result

def _update_paths(paths_dict: dict[str, object], old_base: str, new_base: str) -> None:
    """
    Recursively update paths in the environment dictionary.

    Args:
        paths_dict (dict): Dictionary containing environment paths.
        old_base (str): Old base path to be replaced.
        new_base (str): New base path to replace the old one.

    Returns:
        None
    """
    for key, value in paths_dict.items():
        if isinstance(value, dict):
            _update_paths(value, old_base, new_base)
        elif key == 'path' and isinstance(value, str) and value.startswith(old_base):
            paths_dict[key] = value.replace(old_base, new_base, 1)

def _create_inputs_csv(inp_csv_base_dir: str) -> str:
    """
    Create the input files CSV in the specified directory.

    Args:
        inp_csv_base_dir (str): Directory where the input CSV will be created.

    Returns:
        str: Path to the created CSV file.
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(inp_csv_base_dir):
        logger.error(f"The specified input CSV base directory does not exist: {inp_csv_base_dir}")
        raise FileNotFoundError(f"The specified input CSV base directory does not exist: {inp_csv_base_dir}")
    inp_csv_path = os.path.join(inp_csv_base_dir, RAW_INPUT_FILENAME)
    input_files_df = pd.DataFrame(columns=RAW_INPUT_CSV_COLUMNS)
    input_files_df.to_csv(inp_csv_path, index=False)
    logger.info(f"Input CSV created at: {inp_csv_path}")
    return inp_csv_path

def _check_inputs_csv(inp_csv_path: str) -> bool:
    """
    Check the input CSV for required columns and external files, and warn the user if found.

    Args:
        inp_csv_path (str): Path to the input CSV.

    Returns:
        bool: True if all input files are internal, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Checking input files CSV: {inp_csv_path}")

    inp_csv_base_dir = os.path.dirname(inp_csv_path)

    if not os.path.exists(inp_csv_path):
        logger.warning(f"File {RAW_INPUT_FILENAME} does not exist in {inp_csv_base_dir}.")
        raise FileNotFoundError(f"The file {RAW_INPUT_FILENAME} does not exist in the specified directory: {inp_csv_base_dir}")

    # Read the CSV into a DataFrame
    input_files_df = pd.read_csv(inp_csv_path)

    # Check for missing required columns in the CSV
    missing_cols = [col for col in RAW_INPUT_CSV_COLUMNS if col not in input_files_df.columns]
    if missing_cols:
        logger.error(f"File {RAW_INPUT_FILENAME} is missing required columns: {missing_cols}")
        raise ValueError(f"The file {RAW_INPUT_FILENAME} does not contain the required columns: {missing_cols}")
    
    if input_files_df.empty:
        logger.info(f"The input files CSV {RAW_INPUT_FILENAME} is empty. No input files to check.")
        return True
    else:
        # Identify rows where the file is not internal (i.e., external files)
        external_mask = ~input_files_df.apply(
            lambda row: parse_csv_internal_path_field(row['internal'], row['path'], inp_csv_base_dir), axis=1
        )

        # Warn if any external files are found
        if external_mask.any():
            external_paths = input_files_df.loc[external_mask, 'path'].tolist()
            logger.warning(
                f"Some input files are external to the 'inputs' folder: {external_paths}\n"
                f"You must check the paths of these files in {RAW_INPUT_FILENAME}, and update them, if the path is not correct."
            )
            return False
        else:
            logger.info(f"All input files of {RAW_INPUT_FILENAME} are internal to the 'inputs' folder.")
            return True

# %% === Define the AnalysisEnvironment dataclass ===
@dataclass
class AnalysisEnvironment:
    """Class to store the analysis environment details."""

    # Metadata
    case_name: str
    creator_user: str = field(default_factory=lambda: _retrieve_current_user())
    creator_specs: dict = field(default_factory=lambda: _retrieve_system_specs())
    creation_time: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    last_user: str = field(default_factory=lambda: _retrieve_current_user())
    last_user_specs: dict = field(default_factory=lambda: _retrieve_system_specs())
    last_update: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    app_version: str = field(default_factory=lambda: _retrieve_app_version())
    folders: dict = field(default_factory=lambda: {})
    config: dict = field(default_factory=lambda: copy.deepcopy(ANALYSIS_CONFIGURATION))

    # Dynamic attributes can be added later
    def __post_init__(self):
        # Ensure folders is always a dict (deepcopy for safety)
        if not hasattr(self, 'folders') or self.folders is None or not isinstance(self.folders, dict):
            self.folders = {}
        if 'base' not in self.folders or 'path' not in self.folders['base']:
            self.folders['base'] = {'path': os.getcwd()}  # Set the base path to the current working directory
        # Ensure config is always a dict (deepcopy for safety)
        if not hasattr(self, 'config') or self.config is None or not isinstance(self.config, dict):
            self.config = copy.deepcopy(ANALYSIS_CONFIGURATION)

    # Method to save the environment to a JSON file
    def to_json(
            self, 
            file_path: str
        ) -> None:
        """
        Save the environment to a JSON file, including dynamic attributes.
        
        Args:
            file_path (str): The path to the JSON file.
        """
        self.last_user = _retrieve_current_user()
        self.last_user_specs = _retrieve_system_specs()
        self.last_update = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        self.app_version = _retrieve_app_version()
        
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    # Method to create analysis folders
    def create_folder_structure(
            self,
            base_dir: str
        ) -> None:
        """
        Create the folder structure for the analysis.
        This function creates the main analysis directory and subdirectories.

        Args:
            base_dir (str): Base directory for the analysis. Must already exist.

        Returns:
            None
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting folder structure creation for case: [{self.case_name}]")

        # Check if base_dir is provided and exists
        if not os.path.isdir(base_dir):
            raise ValueError(f"The specified base directory does not exist: [{base_dir}]")

        self.folders['base'] = {'path': base_dir}

        # Dynamically create folder structure attributes
        for key, substructure in ANALYSIS_FOLDER_STRUCTURE.items():
            main_path = os.path.join(base_dir, key)
            nested_dict = _create_nested_folders(main_path, substructure)
            # Set the attribute directly (now declared in dataclass)
            self.folders[key] = nested_dict

        # Create or update the csv of the input files, in the main input folder
        inp_csv_path = _create_inputs_csv(self.folders['inputs']['path'])
        logger.info(f"File {RAW_INPUT_FILENAME} created: {inp_csv_path}")

        # Save the environment to a JSON file
        env_file_path = os.path.join(self.folders['base']['path'], ENVIRONMENT_FILENAME)
        self.to_json(env_file_path)
        logger.info(f"Analysis environment saved to: [{env_file_path}]")

        logger.info(f"Analysis folder structure created successfully for: [{self.case_name}]")
    
    # Method to add an input file to the RAW_INPUT_FILENAME
    def add_input_file(
            self, 
            file_path: str,
            file_type: str,
            file_subtype: str = None,
            force_add: bool = True,
        ) -> tuple[bool, str]:
        """
        Add a new input file to the analysis environment and row in the input files CSV.

        Args:
            file_path (str): Path to the input file to be added.
            file_type (str): Type of the input file (e.g., 'shapefile', 'raster', etc.).
            file_subtype (str, optional): Subtype of the input file (e.g., 'recording', 'forecast', etc.).
            force_add (bool, optional): If True, will overwrite the existing row if it exists (defaults to True).

        Returns:
            bool: True if the file was added successfully, False if it already exists and duplicates are not allowed.
            str: The new ID of the added row.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Adding input file: [{file_path}] (type: [{file_type}])")
        
        inp_csv_path = os.path.join(self.folders['inputs']['path'], RAW_INPUT_FILENAME)
        
        # Check if the input file already exists in the input files CSV
        entry_already_exists = check_raw_path(
            csv_path=inp_csv_path, 
            type=file_type,
            subtype=file_subtype
        )

        row_added, new_id = add_row_to_csv(
                csv_path=inp_csv_path, 
                path_to_add=file_path, 
                path_type=file_type, 
                path_subtype=file_subtype,
                force_rewrite=force_add
            )

        if row_added:
            if entry_already_exists:
                logger.warning(f"File [{file_path}] might already exist in the input files CSV, but it was added/overwritten anyway.")
            else:
                logger.info(f"File [{file_path}] added to the input files CSV.")
        else:
            logger.info(f"File [{file_path}] not added in the input files CSV.")
        return (row_added, new_id)
    
    # Method to save variables in a file
    def save_variable(
            self, 
            variable_to_save: dict[str, object], 
            variable_filename: str,
            environment_filename: str = ENVIRONMENT_FILENAME,
            compression: str = None,
        ) -> None:
        """
        saves variables in a pickle file (optionally compressed) and updates the config dictionary in the environment.
        
        Args:
            variable_to_save (Dict[str, Any]): Dictionary containing the variables to save.
            variable_filename (str): Name of the file to save the variables (must end with '.pkl').
            environment_filename (str): Name of the environment file to update (default is ENVIRONMENT_FILENAME).
            compression (str, optional): Compression type ('gzip' or 'bz2') or None for no compression.
        
        Raises:
            TypeError: If variable_to_save is not a dictionary.
            FileNotFoundError: If the var_dir directory does not exist.
            IOError: If there is an error during file saving.
            ValueError: If compression is not 'gzip', 'bz2', or None.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Start saving variable: [{variable_filename}] (compression: [{compression}])")

        if compression not in [None, 'gzip', 'bz2']:
            logger.error(f"Expected compression to be None, 'gzip', or 'bz2', received: [{compression}]")
            raise ValueError(f"Expected compression to be None, 'gzip', or 'bz2', received: [{compression}]")

        if os.path.isabs(variable_filename):
            variable_filename = os.path.basename(variable_filename)
            logger.warning("filename was provided as an absolute path, converted to filename only.")
        
        # Verify that variable_to_save is a dictionary
        if not isinstance(variable_to_save, dict):
            logger.error(f"variable_to_save must be a dictionary, received: [{type(variable_to_save)}]")
            raise TypeError(f"variable_to_save must be a dictionary, received: [{type(variable_to_save)}]")
        
        # Verify that var_dir exists
        var_dir_path = self.folders['variables']['path']
        if not os.path.exists(var_dir_path):
            logger.error(f"var_dir directory not found: {var_dir_path}")
            raise FileNotFoundError(f"variables directory not found: [{var_dir_path}]")
        
        # Construct the full file path
        file_path = os.path.join(var_dir_path, variable_filename)
        
        # Save the variable to a pickle file, with optional compression
        try:
            if compression == 'gzip':
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(variable_to_save, f)
            elif compression == 'bz2':
                with bz2.open(file_path, 'wb') as f:
                    pickle.dump(variable_to_save, f)
            elif compression is None:
                with open(file_path, 'wb') as f:
                    pickle.dump(variable_to_save, f)
            else:
                logger.error(f"Compression [{compression}] not implemented.")
                raise ValueError(f"Compression [{compression}] not implemented.")
            
            logger.info(f"Variables saved successfully in: [{file_path}] (compression: [{compression}])")
        except Exception as e:
            logger.error(f"Error during file saving [{file_path}]: {e}")
            raise IOError(f"Error during file saving [{file_path}]: {e}")
        
        # Update env.config[filename] with variable labels and compression info
        variable_keys = list(variable_to_save.keys())
        self.config['variables'][variable_filename] = {
            'labels': variable_keys,
            'compression': compression
        }

        if os.path.isabs(environment_filename):
            environment_filename = os.path.basename(environment_filename)
            logger.warning("environment_filename was provided as an absolute path, converted to filename only.")

        env_file_path = os.path.join(self.folders['base']['path'], environment_filename)
        self.to_json(env_file_path)

        logger.info(f"Updated env.config['{variable_filename}'] with {len(variable_keys)} variables (compression: [{compression}])")

    # Method to load variables from file
    def load_variable(
            self,
            variable_filename: str,
        ) -> dict[str, object]:
        """
        Load variables from a pickle file (optionally compressed) and return them as a dictionary.
        
        Args:
            variable_filename (str): name of the file to load, must end with '.pkl' (e.g., 'study_area_vars.pkl').
        
        Returns:
            Dict[str, Any]: dictionary containing the loaded variables.
        
        Raises:
            KeyError: if the variable_filename is not found in env.config['variables'].
            FileNotFoundError: if the file does not exist in the variables directory.
            IOError: if there is an error loading the file.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Start loading variables from [{variable_filename}]")

        if os.path.isabs(variable_filename):
            variable_filename = os.path.basename(variable_filename)
            logger.warning("variable_filename was provided as an absolute path, converted to filename only.")
        
        # Verify that variable_filename is a valid key in env.config['variables']
        available_files = list(self.config['variables'].keys())
        if variable_filename not in available_files:
            logger.error(f"File [{variable_filename}] not found in env.config['variables']. Available files: {available_files}")
            raise KeyError(f"File [{variable_filename}] not found in env.config['variables']. Available files: {available_files}")
        
        # Get compression info
        compression = self.config['variables'][variable_filename].get('compression', None)
        
        # Construct the full file path
        var_dir_path = self.folders['variables']['path']
        file_path = os.path.join(var_dir_path, variable_filename)
        
        # Verify that the file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: [{file_path}]")
            raise FileNotFoundError(f"File does not exist: [{file_path}]")
        
        # Load the variable from the pickle file, with optional decompression
        logger.info(f"Loading variables from: [{file_path}] (compression: [{compression}])")
        try:
            if compression == 'gzip':
                with gzip.open(file_path, 'rb') as f:
                    variable_data = pickle.load(f)
            elif compression == 'bz2':
                with bz2.open(file_path, 'rb') as f:
                    variable_data = pickle.load(f)
            elif compression is None:
                with open(file_path, 'rb') as f:
                    variable_data = pickle.load(f)
            else:
                logger.error(f"Compression [{compression}] not implemented.")
                raise ValueError(f"Compression [{compression}] not implemented.")
            
            if not isinstance(variable_data, dict):
                logger.error(f"Loaded data from [{file_path}] is not a dictionary, instead: [{type(variable_data)}]")
                raise TypeError(f"Loaded data from [{file_path}] is not a dictionary, instead: [{type(variable_data)}]")
            
            logger.info(f"Variables loaded successfully from: [{file_path}]")

            return variable_data
        except Exception as e:
            logger.error(f"Error loading file [{file_path}]: {e}")
            raise IOError(f"Error loading file [{file_path}]: {e}")
    
    # Method to collect input files into analysis folders
    def collect_input_files(
            self,
            file_type: list[str] = None,
            file_subtype: list[str] = None,
            file_custom_id: list[str] = None,
            multi_extension: bool = False
        ) -> None:
        """
        Collect input files from the input files CSV and copy them to the input directory.

        Args:
            file_type (list): List of file types to collect.
            file_subtype (list): List of file subtypes to collect.
            file_custom_id (list): List of file custom IDs to collect.
            multi_extension (bool): Whether to allow multiple file extensions in the input files CSV.

        Returns:
            None
        """
        logger = logging.getLogger(__name__)
        logger.info("Collecting input files from the input files CSV and copying them to the input directory...")

        input_dir = self.folders['inputs']['path']
        inp_csv_path = os.path.join(input_dir, RAW_INPUT_FILENAME)
        if not os.path.exists(inp_csv_path):
            logger.error(f"Input files CSV not found at [{inp_csv_path}]. Please create it first.")
            raise FileNotFoundError(f"Input files CSV not found at [{inp_csv_path}]. Please create it first.")
        
        inp_files_df = pd.read_csv(inp_csv_path)

        # Check if file_custom_id is a list and if all elements are None
        if isinstance(file_custom_id, list) and all([x is None for x in file_custom_id]):
            file_custom_id = None
        if isinstance(file_type, list) and all([x is None for x in file_type]):
            file_type = None
        if isinstance(file_subtype, list) and all([x is None for x in file_subtype]):
            file_subtype = None

        # Filter the DataFrame based on the provided file_type, file_subtype, and file_custom_id
        if file_custom_id:
            if not isinstance(file_custom_id, list):
                logger.error("file_id must be a list of strings.")
                raise TypeError("file_id must be a list of strings.")
            
            if file_type or file_subtype:
                logger.error("file_type and file_subtype cannot be used with file_custom_id.")
                raise ValueError("file_type and file_subtype cannot be used with file_custom_id.")
            
            df_filter = inp_files_df['custom_id'].isin(file_custom_id)
        else:
            df_filter = np.ones(len(inp_files_df), dtype=bool)

            if file_type:
                if not isinstance(file_type, list):
                    logger.error("file_type must be a list of strings.")
                    raise TypeError("file_type must be a list of strings.")
                df_filter = df_filter & inp_files_df['type'].isin(file_type)
            
            if file_subtype:
                if not isinstance(file_subtype, list):
                    logger.error("file_subtype must be a list of strings.")
                    raise TypeError("file_subtype must be a list of strings.")
                df_filter = df_filter & inp_files_df['subtype'].isin(file_subtype)
        
        inp_files_df_filtered = inp_files_df[df_filter]
        
        for idx, row in inp_files_df_filtered.iterrows():
            # Parse the internal path field to get the correct file(s) path
            if parse_csv_internal_path_field(row['internal'], row['path'], inp_csv_path):
                logger.info(f"File [{row['path']}] already exists in the input directory. Skipping copy.")
                continue
            if multi_extension:
                # If multi_extension is True, we need to find all files with the same basename
                file_basename_no_ext = os.path.splitext(os.path.basename(row['path']))[0] # Also with multiple ".", splitext will split just the last "."
                file_paths = [os.path.join(os.path.dirname(row['path']), file) 
                              for file in os.listdir(os.path.dirname(row['path'])) 
                              if file.startswith(file_basename_no_ext)]
            else:
                file_paths = [row['path']] # Just use the single path

            file_type = row['type']
            known_types = self.folders['inputs'].keys()
            if file_type in known_types:
                rel_inp_dir = self.folders['inputs'][file_type]['path']
            else:
                rel_inp_dir = self.folders['inputs'][GENERIC_INPUT_TYPE[0]]['path']

            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File [{file_path}] does not exist. Skipping...")
                    continue
                
                # Check if the file is already in the input directory
                dest_path = os.path.join(rel_inp_dir, os.path.basename(file_path))
                
                if not os.path.exists(dest_path):
                    try:
                        # Copy the file to the input directory
                        shutil.copy(file_path, dest_path)
                        logger.info(f"Copied [{file_path}] to [{dest_path}]")
                    except Exception as e:
                        logger.error(f"Error copying file [{file_path}] to [{dest_path}]: {e}")
                        raise IOError(f"Error copying file [{file_path}] to [{dest_path}]: {e}")
                else:
                    logger.info(f"File {file_path} already exists in the input directory. Skipping copy.")
            
            # Update the input files DataFrame
            main_file = os.path.join(rel_inp_dir, os.path.basename(row['path']))
            inp_files_df_filtered.loc[idx, ['path', 'internal']] = [os.path.relpath(main_file, input_dir), True]

        # Save the updated input files DataFrame
        inp_files_df.loc[df_filter, ['path', 'internal']] = inp_files_df_filtered[['path', 'internal']]
        inp_files_df.to_csv(inp_csv_path, index=False)
        logger.info("Input files collected and copied to the input directory successfully.")
    
    # Method to generate default CSV files
    def generate_default_csv(self) -> None:
        """Generate default CSV files in the user control directory."""
        logger = logging.getLogger(__name__)
        logger.info("Generating default CSV files in the user control directory...")

        def_std_cls_filename = os.path.join(self.folders['user_control']['path'], STANDARD_CLASSES_FILENAME)
        pd.DataFrame(DEFAULT_STANDARD_CLASSES).to_csv(
            def_std_cls_filename,
            index=False
        )
        logger.info(f"Default standard classes CSV generated at: [{def_std_cls_filename}]")

        def_par_cls_filename = os.path.join(self.folders['user_control']['path'], PARAMETER_CLASSES_FILENAME)
        pd.DataFrame(DEFAULT_PARAMETER_CLASSES).to_csv(
            def_par_cls_filename,
            index=False
        )
        logger.info(f"Default parameter classes CSV generated at: [{def_par_cls_filename}]")

        def_ref_pts_filename = os.path.join(self.folders['user_control']['path'], REFERENCE_POINTS_FILENAME)
        pd.DataFrame(columns=REFERENCE_POINTS_CVS_COLUMNS).to_csv(
            def_ref_pts_filename,
            index=False
        )
        logger.info(f"Default reference points CSV generated at: [{def_ref_pts_filename}]")

    # Method to load the environment from a JSON file
    @classmethod
    def from_json(cls, file_path: str) -> "AnalysisEnvironment":
        """Load the environment from a JSON file, including dynamic attributes."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Extract only the fields that are in the dataclass signature for __init__
        init_fields = {f.name for f in cls.__dataclass_fields__.values()}
        init_args = {k: v for k, v in data.items() if k in init_fields}
        obj = cls(**init_args)
        # Set any extra fields (dynamic attributes)
        for k, v in data.items():
            if k not in init_fields:
                setattr(obj, k, v)
        return obj

# %% === Methods to create or get the analysis environment ===
def create_analysis_environment(base_dir: str, case_name: str = None) -> AnalysisEnvironment:
    """
    Create a new analysis and its folder structure.

    Args:
        base_dir (str): Base directory for the analysis. Must exist.
        case_name (str, optional): Name of the case study. If None, the default is used.

    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """
    _check_libraries()

    logger = logging.getLogger(__name__)
    logger.info(f"Start creating environment in: {base_dir}")

    if not os.path.isdir(base_dir):
        raise ValueError(f"The specified base directory does not exist: {base_dir}")

    if os.path.exists(os.path.join(base_dir, ENVIRONMENT_FILENAME)):
        error_msg = f"An analysis environment already exists in the specified base directory: {base_dir}"
        logger.error(error_msg)
        raise FileExistsError(
            error_msg + " Please choose a different base directory or delete the existing environment."
        )

    if case_name is None:
        case_name = DEFAULT_CASE_NAME
    
    case_name_clean = re.sub(r'[\\/:*?"<>| ]', '_', case_name)  # Replace spaces and invalid filename characters with underscores
    
    # Create the analysis environment object
    env = AnalysisEnvironment(
        case_name=case_name_clean
    )

    env.create_folder_structure(base_dir)

    env.generate_default_csv()

    # Set up the session logger
    current_datetime = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = os.path.join(env.folders['logs']['path'], f"{env.last_user}_{env.case_name}_session_{current_datetime}.log")
    _setup_session_logger(log_file_path)

    return env

def get_analysis_environment(base_dir: str) -> AnalysisEnvironment:
    """
    Load an existing analysis environment from the specified directory.

    Args:
        base_dir (str): Base directory of the analysis. Must exist.

    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """
    _check_libraries()

    logger = logging.getLogger(__name__)
    logger.info(f"Start loading environment from: {base_dir}")

    if base_dir is None:
        raise ValueError("base_dir must be provided to load an existing analysis environment.")

    if not os.path.isdir(base_dir):
        raise ValueError(f"The specified base directory does not exist: {base_dir}")

    env_file_path = os.path.join(base_dir, ENVIRONMENT_FILENAME)
    if os.path.exists(env_file_path):
        logger.info(f"Existing {ENVIRONMENT_FILENAME} found in {base_dir}. Loading environment...")

        env = AnalysisEnvironment.from_json(env_file_path)
        old_base = env.folders['base']['path']
        
        # If the base directory has changed, update all paths accordingly
        if os.path.abspath(old_base) != os.path.abspath(base_dir):
            logger.warning(f"Base directory has changed from '{old_base}' to '{base_dir}'. Updating all paths...")
            # Recursively update all paths in the environment for each folder attribute
            for key, val in env.folders.items():
                if isinstance(val, dict):
                    _update_paths(val, old_base, base_dir)
                else:
                    logger.error(f"Unexpected type for attribute 'folders.{key}': {type(val)}. Expected dict.")
                    raise TypeError(f"Expected dict for attribute 'folders.{key}', got {type(val)}.")
                
            # Save the updated environment to file
            env.to_json(env_file_path)
            logger.info("All paths updated and environment saved.")

            inp_csv_base_dir = env.folders['inputs']['path']
            inp_csv_path = os.path.join(inp_csv_base_dir, RAW_INPUT_FILENAME)
            # If the input CSV does not exist, create it; otherwise, check its contents
            if not os.path.exists(inp_csv_path):
                logger.warning(f"File {RAW_INPUT_FILENAME} does not exist in {inp_csv_base_dir}. Creating it...")
                _create_inputs_csv(inp_csv_base_dir)
                logger.info(f"File {RAW_INPUT_FILENAME} created: {inp_csv_path}")
            else:
                logger.info(f"File {RAW_INPUT_FILENAME} already exists: {inp_csv_path}")
                _check_inputs_csv(inp_csv_path)

        else:
            logger.info("Base directory unchanged. Environment loaded as is.")

        # Set up a session log file
        current_datetime = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_path = os.path.join(env.folders['logs']['path'], f"{env.last_user}_{env.case_name}_session_{current_datetime}.log")
        _setup_session_logger(log_file_path)

        return env
    
    else:
        raise FileNotFoundError(f"No existing analysis environment found in {base_dir}.")
    
# %%
