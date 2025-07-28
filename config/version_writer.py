#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for writing the application version to a file.
This module provides functions to get the current application version.
"""

# %% Import necessary libraries
import subprocess
import os

# %% Define functions to get the version and write it to a file
def get_app_version():
    try:
        output = subprocess.check_output(["git", "describe", "--tags", "--always", "--dirty"])
        version = output.decode("utf-8").strip()
        return version
    except subprocess.CalledProcessError as e:
        print(f"Error getting version: {e}")
        return None

def write_app_version_to_file(version):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "version.txt")
    with open(file_path, "w") as f:
        f.write(version)
    return file_path

# %% Main execution
if __name__ == "__main__":
    version = get_app_version()
    if version:
        file_path = write_app_version_to_file(version)
        print(f"Wrote version in {file_path}: {version}")