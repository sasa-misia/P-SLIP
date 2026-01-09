# %% === Import necessary modules
import argparse
import logging
import os
import sys
import warnings

try:
    import psutil
    SYSTEM_MONITORING = True
except ImportError:
    SYSTEM_MONITORING = False

# Add the parent directory to the system path (temporarily)
# This allows importing modules from the parent directory (like config and psliptools)
# This is necessary for the script to run correctly when executed directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.analysis_init import create_analysis_environment, get_analysis_environment, AnalysisEnvironment
from config.default_params import LOG_CONFIG, ENVIRONMENT_FILENAME, DEFAULT_CASE_NAME

# %% === Logger
# This will log messages to the console and when the AnalysisEnvironment is created or loaded, it will log to a file
def setup_logger(
        module_name=None
    ):
    """
    Set up the logger for the analysis environment.

    Args:
        module_name: Name of the module (use __name__). If None, the module where this method is called is used.

    Returns:
        logging.Logger: The logger object.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_CONFIG['format'], 
        datefmt=LOG_CONFIG['date_format']
    )
    if module_name:
        logger = logging.getLogger(module_name)
    else:
        logger = logging.getLogger(__name__)
    return logger

# This will log warnings and raise warnings, using the given logger or creating a new one
def log_and_warning(
        warning_msg: str, 
        stacklevel: int=2,
        logger: logging.Logger=None
    ) -> None:
    """
    Log a warning message and raise a warning.

    Args:
        warning_msg: Warning message to log.
        logger: Logger object. If None, a new logger is created.
    """
    if logger is None:
        logger = setup_logger()
    
    logger.warning(warning_msg)

    warnings.warn(warning_msg, stacklevel=stacklevel)

# This will log errors and raise exceptions, using the given logger or creating a new one
def log_and_error(
        error_msg: str, 
        exception_type: type=ValueError, 
        logger: logging.Logger=None
    ) -> None:
    """
    Log an error message and raise the specified exception.

    Args:
        error_msg: Error message to log.
        exception_type: Exception type to raise.
        logger: Logger object. If None, a new logger is created.
    """
    if logger is None:
        logger = setup_logger()
    
    logger.error(error_msg)

    raise exception_type(error_msg)

# This will log memory usage, using the given logger or creating a new one
def memory_report(
        logger: logging.Logger=None
    ) -> None:
    """
    Report memory usage.

    Args:
        logger: Logger object. If None, a new logger is created.
    """
    if logger is None:
        logger = setup_logger()
    
    if SYSTEM_MONITORING:
        mem = psutil.virtual_memory()
        logger.info(f"Memory usage: {mem.percent}% ({mem.used / 1024**3:.2f} GB used) ...")
    else:
        logger.info("Memory monitoring not available...")

# This will return hardware information
def get_hardware_info() -> dict[str, int | float]:
    """
    Report hardware information.

    Returns:
        dict(str, int | float): Dictionary with hardware information (if available). Keys are 'cores', 'threads', 'ram', 'disk', 'free_disk', 'free_ram'.
    """
    if SYSTEM_MONITORING:
        try:
            disk_usage = psutil.disk_usage(os.getcwd())
            disk_total = disk_usage.total / 1024**3
            disk_free = disk_usage.free / 1024**3
        except Exception:
            disk_total = 0
            disk_free = 0
        
        hardware_info = {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'ram': psutil.virtual_memory().total / 1024**3,
            'disk': disk_total,
            'free_disk': disk_free,
            'free_ram': psutil.virtual_memory().available / 1024**3
        }
    else:
        hardware_info = { # If SYSTEM_MONITORING is False, return fake default values (zero)
            'cores': 0,
            'threads': 0,
            'ram': 0,
            'disk': 0,
            'free_disk': 0,
            'free_ram': 0
        }

    return hardware_info

# This will log hardware information, using the given logger or creating a new one
def hardware_report(
        logger: logging.Logger=None
    ) -> None:
    """
    Report hardware information.

    Args:
        logger: Logger object. If None, a new logger is created.
    """
    hardware_info = get_hardware_info()

    if logger is None:
        logger = setup_logger()
    
    logger.info(f"Hardware information: {hardware_info}")

# %% === Main function to initialize or load the analysis environment
# This function is responsible for creating or loading the analysis environment based on user input.
def get_or_create_analysis_environment(
        base_dir=None,
        gui_mode=False,
        case_name=None,
        allow_creation=True,
        env_filename=None
    ) -> AnalysisEnvironment:
    """
    Main function for initializing or loading the analysis environment.

    Args:
        case_name: Case study name. If None, a default name is used.
        gui_mode: If true, the function was called from a GUI.
        base_dir: Base directory for the analysis. If None, the current directory is used.

    Returns:
        AnalysisEnvironment: Object with the details of the analysis environment.
    """

    # Prompt for case name and base directory if not provided
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        if base_dir is None:
            base_dir = input(f"Enter the base directory for the analysis (or press Enter to use the current directory [{os.getcwd()}]): ").strip(' "')
            if not base_dir:
                base_dir = os.getcwd()

    os.makedirs(base_dir, exist_ok=True) # Ensure the base directory exists

    if env_filename:
        if not isinstance(env_filename, str):
            raise ValueError("env_file_path must be a string representing the path to the environment file.")
        if not env_filename.endswith('.json'):
            raise ValueError("env_file_path must be a JSON file.")
        if os.path.isabs(env_filename):
            env_filename = os.path.basename(env_filename)
        CURR_ENV_FILENAME = env_filename
    else:
        CURR_ENV_FILENAME = ENVIRONMENT_FILENAME

    # Decide automatically based on base_dir existence
    if os.path.exists(os.path.join(base_dir, CURR_ENV_FILENAME)):
        env = get_analysis_environment(base_dir=base_dir)
    else:
        if allow_creation:
            if case_name is None:
                if gui_mode:
                    raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
                else:
                    case_name = input("Specify the analysis name (enter for default [Not Defined - Standalone]): ").strip()
                    if not case_name:
                        case_name = DEFAULT_CASE_NAME
            env = create_analysis_environment(base_dir=base_dir, case_name=case_name)
        else:
            raise FileNotFoundError(f"Analysis environment file '{CURR_ENV_FILENAME}' not found in {base_dir}. "
                                     "Please create the environment first or set allow_creation=True.")

    logging.info(f"Analysis environment loaded for case: {env.case_name}")
    return env

# %% Utuility functions
def obtain_config_idx_and_rel_filename(
        env: AnalysisEnvironment, 
        source_type: str, 
        source_subtype: str=None
    ) -> tuple[AnalysisEnvironment, int, str]:
    """
    Utility function to obtain the config index and relative filename based on the source type and subtype.

    Args:
        env (AnalysisEnvironment): The analysis environment object.
        source_type (str): The source type.
        source_subtype (str, optional): The source subtype. Defaults to None.

    Returns:
        tuple[AnalysisEnvironment, int, str]: The modified analysis environment, the config index, and the possible relative filename to use for output.
    """
    idx = 0
    if source_subtype:
        if env.config['inputs'][source_type][0]['settings']: # if the setting dictionary of the first element [0] is not empty, then you should overwrite or add an element to the list
            poss_idx = []
            for i, d in enumerate(env.config['inputs'][source_type]):
                if 'source_subtype' in d['settings'].keys():
                    if d['settings']['source_subtype'] == source_subtype:
                        poss_idx.append(i)
            
            if len(poss_idx) > 1:
                raise ValueError("Multiple subtypes with the same name were found. Please check the subtype.")
            elif len(poss_idx) ==  1:
                idx = poss_idx[0]
            else:
                idx += len(env.config['inputs'][source_type]) # This must be before the append!
                env.config['inputs'][source_type].append({})
            
        rel_filename = f"{source_type}_{source_subtype}"

    else:
        rel_filename = f"{source_type}"
        
    return env, idx, rel_filename

# %% === Command line interface
# This block allows the script to be run from the command line with parameters.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or load the folder structure for the analysis.")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for the analysis")
    parser.add_argument("--gui_mode", action="store_true", help="Run in GUI mode (not implemented yet)")
    parser.add_argument("--case_name", type=str, default=None, help="Case study name")
    parser.add_argument("--allow_creation", action="store_true", help="Allow creation of the environment if it doesn't exist")
    parser.add_argument("--env_filename", type=str, default=None, help="Environment filename")
    args = parser.parse_args()

    # Call the main function with the provided arguments
    curr_env = get_or_create_analysis_environment(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode,
        case_name=args.case_name,
        allow_creation=args.allow_creation,
        env_filename=args.env_filename
    )

# %%
