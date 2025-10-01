"""
Scattered utilities for the psliptools package.

Provides functions for importing scattered data, typically from CSV files of time-sensitive data, such as rainfall.
"""

from .import_scattered import (
    load_time_sensitive_data_from_csv,
    load_time_sensitive_gauges_from_csv,
    merge_time_sensitive_data_with_gauges
)

__all__ = [
    "load_time_sensitive_data_from_csv",
    "load_time_sensitive_gauges_from_csv",
    "merge_time_sensitive_data_with_gauges"
]