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
    DEFAULT_PLOT_CONFIG,
    LOG_CONFIG,
    RAW_INPUT_FILENAME,
    RAW_INPUT_CSV_COLUMNS,
    LIBRARIES_CONFIG,
    ANALYSIS_CONFIGURATION
)

from .version_writer import (
    get_app_version,
    write_app_version_to_file
)

from .analysis_init import (
    AnalysisEnvironment,
    create_analysis_environment,
    get_analysis_environment,
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
    'get_app_version',
    'write_app_version_to_file'
]