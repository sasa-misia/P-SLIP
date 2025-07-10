#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Installation file for package config.
"""

from .default_params import (
    DEFAULT_CASE_NAME,
    ENVIRONMENT_FILENAME,
    GENERIC_INPUT_TYPE,
    ANALYSIS_FOLDER_STRUCTURE,
    ANALYSIS_FOLDER_ATTRIBUTE_MAPPER,
    DEFAULT_PLOT_CONFIG,
    LOG_CONFIG,
    RAW_INPUT_FILENAME,
    RAW_INPUT_CSV_COLUMNS,
    LIBRARIES_CONFIG,
    ANALYSIS_CONFIGURATION
)

from .analysis_init import (
    AnalysisEnvironment,
    create_analysis_environment,
    get_analysis_environment,
    add_input_file,
    save_variable,
    load_variable
)

__all__ = [
    'DEFAULT_CASE_NAME',
    'ENVIRONMENT_FILENAME',
    'GENERIC_INPUT_TYPE',
    'ANALYSIS_FOLDER_STRUCTURE',
    'ANALYSIS_FOLDER_ATTRIBUTE_MAPPER',
    'DEFAULT_PLOT_CONFIG',
    'LOG_CONFIG',
    'RAW_INPUT_FILENAME',
    'RAW_INPUT_CSV_COLUMNS',
    'LIBRARIES_CONFIG',
    'ANALYSIS_CONFIGURATION',
    'AnalysisEnvironment',
    'create_analysis_environment',
    'get_analysis_environment',
    'add_input_file',
    'save_variable',
    'load_variable'
]