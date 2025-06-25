#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Installation file for package config.
"""

from .default_params import (
    DEFAULT_CASE_NAME,
    ENVIRONMENT_FILENAME,
    ANALYSIS_FOLDER_STRUCTURE,
    ANALYSIS_FOLDER_ATTRIBUTE_MAPPER,
    PLOT_CONFIG,
    LOG_CONFIG,
    RAW_INPUT_FILENAME,
    RAW_INPUT_CSV_COLUMNS,
    LIBRARIES_CONFIG
)

from .analysis_init import (
    create_analysis_environment,
    get_analysis_environment,
)

__all__ = [
    'DEFAULT_CASE_NAME',
    'ENVIRONMENT_FILENAME',
    'ANALYSIS_FOLDER_STRUCTURE',
    'ANALYSIS_FOLDER_ATTRIBUTE_MAPPER',
    'PLOT_CONFIG',
    'LOG_CONFIG',
    'RAW_INPUT_FILENAME',
    'RAW_INPUT_CSV_COLUMNS',
    'LIBRARIES_CONFIG',
    'create_analysis_environment',
    'get_analysis_environment'
]