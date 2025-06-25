#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for creating the folder structure for the analysis.
"""

import os
import platform
import importlib
import warnings
import json
import logging
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple

# Import default configuration
from config import (
    RAW_INPUT_FILENAME,
    RAW_INPUT_CSV_COLUMNS,
    ENVIRONMENT_FILENAME,
    ANALYSIS_FOLDER_STRUCTURE,
    ANALYSIS_FOLDER_ATTRIBUTE_MAPPER,
    LIBRARIES_CONFIG,
    DEFAULT_CASE_NAME,
)

@dataclass
class AnalysisEnvironment:
    """Class to store the analysis environment details."""

    # Metadata
    case_name: str
    user: str
    os_separator: str
    creation_time: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    base_dir: Dict[str, str] = field(default_factory=lambda: {'path': os.getcwd()})

    def __post_init__(self):
        # Dynamically ensure all folder attributes exist and are dicts
        for key, attr in ANALYSIS_FOLDER_ATTRIBUTE_MAPPER.items():
            if not hasattr(self, attr) or getattr(self, attr) is None:
                setattr(self, attr, {})
            elif not isinstance(getattr(self, attr), dict):
                setattr(self, attr, {})

    def to_json(self, file_path: str) -> None:
        """Save the environment to a JSON file, including dynamic attributes."""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

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
            "scikit-image": "skimage",
            "scikit-learn": "sklearn",
        }
        if corrected_name in mapping:
            importlib.import_module(mapping[corrected_name])
            return mapping[corrected_name]
        # If it still fails, raise an error
        raise ImportError(
            f"Library '{lib}' could not be imported. Please ensure it is installed correctly or add entry to the mapping (eg: scikit-image: skimage)."
        )

def _check_libraries(required_file: str = LIBRARIES_CONFIG['required_file'],
                     optional_file: str = LIBRARIES_CONFIG['optional_file']) -> Tuple[bool, bool]:
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
        error_msg = "The following required libraries must be installed:\n" + "\n".join(missing_required)
        logger.error(error_msg)
        raise ImportError(error_msg)

    if missing_optional:
        warn_msg = "It is recommended to install the following optional libraries:\n" + "\n".join(missing_optional)
        logger.warning(warn_msg)
        warnings.warn(warn_msg)

    logger.info("Library check completed: required OK=%s, optional OK=%s", len(missing_required) == 0, len(missing_optional) == 0)
    return (len(missing_required) == 0, len(missing_optional) == 0)

def _create_nested_folders(base_path: str, structure: list) -> Dict[str, Any]:
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
                result[subkey] = _create_nested_folders(sub_path, subval)
    return result

def _update_paths(env_dict: Dict[str, Any], old_base: str, new_base: str) -> None:
    """
    Recursively update paths in the environment dictionary.

    Args:
        env_dict (dict): Dictionary containing environment paths.
        old_base (str): Old base path to be replaced.
        new_base (str): New base path to replace the old one.

    Returns:
        None
    """
    for key, value in env_dict.items():
        if isinstance(value, dict):
            _update_paths(value, old_base, new_base)
        elif key == 'path' and isinstance(value, str) and value.startswith(old_base):
            env_dict[key] = value.replace(old_base, new_base, 1)

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

def _is_relative_path(raw_inp_path: str, raw_inp_base_dir: str) -> bool:
    """
    Check if the given path is relative to the base directory.

    Args:
        raw_inp_path (str): Input file path.
        raw_inp_base_dir (str): Base directory.

    Returns:
        bool: True if the path is relative to the base directory, False otherwise.
    """
    abs_pth = os.path.abspath(os.path.join(raw_inp_base_dir, raw_inp_path)) if not os.path.isabs(raw_inp_path) else raw_inp_path
    abs_inp = os.path.abspath(raw_inp_base_dir)
    return abs_pth.startswith(abs_inp)

def _parse_internal(value: str | int | float, raw_inp_path: str, raw_inp_base_dir: str) -> bool:
    """
    Parse the 'internal' field from the input CSV.

    Args:
        value (str|int|float): Value from the 'internal' column.
        raw_inp_path (str): Path from the CSV.
        raw_inp_base_dir (str): Base directory.

    Returns:
        bool: True if internal, False otherwise.
    """
    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in ("true", "1"):
            return True
        elif val_lower in ("false", "0"):
            return False
    elif isinstance(value, (int, float)):
        if value in (1, 0):
            return bool(value)
    # If value is missing or not recognized, fallback to path check
    return _is_relative_path(raw_inp_path, raw_inp_base_dir)

def _check_inputs_csv(inp_csv_path: str, inp_csv_base_dir: str) -> bool:
    """
    Check the input CSV for required columns and external files, and warn the user if found.

    Args:
        inp_csv_path (str): Path to the input CSV.
        inp_csv_base_dir (str): Base directory for the input CSV.

    Returns:
        bool: True if all input files are internal, False otherwise.
    """
    # Get a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Checking input files CSV: {inp_csv_path}")

    # Check if the input CSV exists
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

    # Identify rows where the file is not internal (i.e., external files)
    external_mask = ~input_files_df.apply(
        lambda row: _parse_internal(row['internal'], row['path'], inp_csv_base_dir), axis=1
    )

    # Warn if any external files are found
    if external_mask.any():
        external_paths = input_files_df.loc[external_mask, 'path'].tolist()
        logger.warning(
            f"Some input files are external to the inputs folder: {external_paths}\n"
            f"You must check the paths of these files in {RAW_INPUT_FILENAME}, and update them if the path is not correct."
        )
        return False
    else:
        logger.info(f"All input files of {RAW_INPUT_FILENAME} are internal to the inputs folder.")
        return True

def _create_folder_structure(base_dir: str, case_name: str) -> AnalysisEnvironment:
    """
    Create the folder structure for the analysis.
    This function creates the main analysis directory and subdirectories.

    Args:
        base_dir (str): Base directory for the analysis. Must exist.
        case_name (str): Name of the case study.

    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting folder structure creation for case: {case_name}")

    # OS detection
    if platform.system() == 'Windows':
        user = os.environ.get('USERNAME', 'unknown')
    else:
        user = os.environ.get('USER', 'unknown')
        if platform.system() == 'Darwin':
            pass
        else:
            logger.warning('Platform not yet tested. In case of problems, please contact the developer.')

    # Separator
    sl = os.path.sep

    # Library check
    logger.info("Checking required and optional libraries...")
    _check_libraries()
    logger.info("Library check completed.")

    # Create the analysis environment object
    env = AnalysisEnvironment(
        case_name=case_name,
        user=user,
        os_separator=sl
    )

    # Check if base_dir is provided and exists
    if not os.path.isdir(base_dir):
        raise ValueError(f"The specified base directory does not exist: {base_dir}")

    env.base_dir = {'path': base_dir}

    # Dynamically create folder structure attributes
    for key, attr in ANALYSIS_FOLDER_ATTRIBUTE_MAPPER.items():
        main_path = os.path.join(base_dir, key)
        structure = ANALYSIS_FOLDER_STRUCTURE.get(key, [])
        nested_dict = _create_nested_folders(main_path, structure)
        # Set the attribute directly (now declared in dataclass)
        setattr(env, attr, nested_dict)

    # Create or update the csv of the input files, in the main input folder
    inp_csv_path = _create_inputs_csv(env.inp_dir['path'])
    logger.info(f"File {RAW_INPUT_FILENAME} created: {inp_csv_path}")

    # Save the environment to a JSON file
    env_file_path = os.path.join(env.base_dir['path'], ENVIRONMENT_FILENAME)
    env.to_json(env_file_path)
    logger.info(f"Analysis environment saved to: {env_file_path}")

    logger.info(f"Analysis folder structure created successfully for: {case_name}")
    return env

def create_analysis_environment(base_dir: str, case_name: Optional[str] = None) -> AnalysisEnvironment:
    """
    Create a new analysis and its folder structure.

    Args:
        base_dir (str): Base directory for the analysis. Must exist.
        case_name (str, optional): Name of the case study. If None, the default is used.

    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """
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

    return _create_folder_structure(base_dir=base_dir, case_name=case_name)

def get_analysis_environment(base_dir: str) -> AnalysisEnvironment:
    """
    Load an existing analysis environment from the specified directory.

    Args:
        base_dir (str): Base directory of the analysis. Must exist.

    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """
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
        old_base = env.base_dir['path']
        # If the base directory has changed, update all paths accordingly
        if os.path.abspath(old_base) != os.path.abspath(base_dir):
            logger.warning(f"Base directory has changed from '{old_base}' to '{base_dir}'. Updating all paths...")
            env.base_dir['path'] = base_dir
            # Recursively update all paths in the environment for each folder attribute
            for attr in ANALYSIS_FOLDER_ATTRIBUTE_MAPPER.values():
                val = getattr(env, attr)
                if isinstance(val, dict):
                    _update_paths(val, old_base, base_dir)
                else:
                    logger.error(f"Unexpected type for attribute '{attr}': {type(val)}. Expected dict.")
                    raise TypeError(f"Expected dict for attribute '{attr}', got {type(val)}.")
            # Save the updated environment to file
            env.to_json(env_file_path)
            logger.info("All paths updated and environment saved.")

            inp_csv_base_dir = env.inp_dir['path']
            inp_csv_path = os.path.join(inp_csv_base_dir, RAW_INPUT_FILENAME)
            # If the input CSV does not exist, create it; otherwise, check its contents
            if not os.path.exists(inp_csv_path):
                logger.warning(f"File {RAW_INPUT_FILENAME} does not exist in {inp_csv_base_dir}. Creating it...")
                _create_inputs_csv(inp_csv_base_dir)
                logger.info(f"File {RAW_INPUT_FILENAME} created: {inp_csv_path}")
            else:
                logger.info(f"File {RAW_INPUT_FILENAME} already exists: {inp_csv_path}")
                _check_inputs_csv(inp_csv_path, inp_csv_base_dir)
        else:
            logger.info("Base directory unchanged. Environment loaded as is.")
        return env
    else:
        raise FileNotFoundError(f"No existing analysis environment found in {base_dir}.")