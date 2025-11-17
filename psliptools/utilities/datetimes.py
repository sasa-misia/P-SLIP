# %% === Import necessary modules
import re
import pandas as pd
import numpy as np
from dateutil import parser

# %% === Global variables
DATE_SEP_CHARS = r'[-/.]' # Possible characters to separate date are [-], [/], or [.]
TIME_SEP_CHARS = r'[:.]' # Possible characters to separate time are [:], or [.]

DATE_PATTERN = re.compile(rf'(\d{{4}}{DATE_SEP_CHARS}\d{{1,2}}{DATE_SEP_CHARS}\d{{1,2}}|\d{{1,2}}{DATE_SEP_CHARS}\d{{1,2}}{DATE_SEP_CHARS}\d{{4}}|\d{{8}})')
TIME_PATTERN = re.compile(rf'(\d{{1,2}}{TIME_SEP_CHARS}\d{{1,2}}{TIME_SEP_CHARS}\d{{1,2}}|\d{{1,2}}{TIME_SEP_CHARS}\d{{1,2}})')

DATETIME_NOT_FOUND_STRING = "Unable to infer date format"

# %% === Helper method to clean strings, based on a datetime pattern (just the pattern will be extracted)
def _clean_date_string(
        date_str: str,
        date_time_separator: str=None
    ) -> str:
    """
    Clean a date string based on a datetime pattern.
    
    Args:
        date_str (str): The date string to clean.
        date_time_separator (str, optional): The separator between date and time (default: None, which means that the existing one will be used)
        
    Returns:
        str: The cleaned date string.
    """
    date_match = DATE_PATTERN.search(date_str)
    time_match = TIME_PATTERN.search(date_str)

    if date_match:
        date_start_idx = date_match.start()
        date_end_idx = date_match.end()
    else:
        raise ValueError(f'No date found in {date_str}')
    
    if time_match:
        time_start_idx = time_match.start()
        time_end_idx = time_match.end()
        
        if date_start_idx < time_start_idx:
            first_part = date_str[date_start_idx:date_end_idx]
            second_part = date_str[time_start_idx:time_end_idx]
            if date_time_separator is None:
                date_time_separator = date_str[date_end_idx:time_start_idx]
        else:
            first_part = date_str[time_start_idx:time_end_idx]
            second_part = date_str[date_start_idx:date_end_idx]
            if date_time_separator is None:
                date_time_separator = date_str[time_end_idx:date_start_idx]
        
        date_str_clean = first_part + date_time_separator + second_part
    else:
        date_str_clean = date_str[date_start_idx:date_end_idx]
        
    return date_str_clean

# %% === Helper method to get compatible date formats
def _get_compatible_date_formats(
        date_str: str
    ) -> list[str]:
    """
    Get a list of compatible date formats from a date string (just the day part).
    
    Args:
        date_str (str): The date string (just the day part) to get compatible datetime formats from.
        
    Returns:
        list[str]: A list of compatible date formats (just the day part, e.g. %d%m%Y).
    """
    date_sep = [x for x in date_str if not x.isdigit()]
    if len(date_sep) == 0 and len(date_str) == 8: # No separators
        if int(date_str[:4]) > 1900 and 1 <= int(date_str[4:6]) <= 31 and 1 <= int(date_str[6:]) <= 31: # Probably starts with year
            if date_str[4:6] > '12': # Day is in the middle
                date_formats = ['%Y%d%m']
            elif date_str[6:] > '12': # Day is at the end
                date_formats = ['%Y%m%d']
            else: # Not sure if day is in the middle or at the end, include both
                date_formats = [
                    '%Y%d%m',
                    '%Y%m%d'
                ]
        elif int(date_str[4:]) > 1900 and 1 <= int(date_str[0:2]) <= 31 and 1 <= int(date_str[2:4]) <= 31: # probably ends with year
            if date_str[0:2] > '12': # Day is at the beginning
                date_formats = ['%d%m%Y']
            elif date_str[2:4] > '12': # Day is in the middle
                date_formats = ['%m%d%Y']
            else: # Not sure if day is at the beginning or in the middle, include both
                date_formats = [
                    '%d%m%Y',
                    '%m%d%Y'
                ]
        else:
            raise ValueError(f"{DATETIME_NOT_FOUND_STRING}. Date string with 8 characters: {date_str}")
    
    elif len(date_sep) == 2 and len(date_str) == 10: # Separators (which can be different, in case of user inconsistency)
        date_splitted = date_str.replace(date_sep[1], date_sep[0]).split(date_sep[0])
        if len(date_splitted[0]) == 4: # Starts with year
            if int(date_splitted[1]) > 12: # Day is in the middle
                date_formats = [f'%Y{date_sep[0]}%d{date_sep[1]}%m']
            elif int(date_splitted[2]) > 12: # Day is at the end
                date_formats = [f'%Y{date_sep[0]}%m{date_sep[1]}%d']
            else: # Not sure if day is in the middle or at the end, include both
                date_formats = [
                    f'%Y{date_sep[0]}%d{date_sep[1]}%m',
                    f'%Y{date_sep[0]}%m{date_sep[1]}%d'
                ]
        elif len(date_splitted[2]) == 4: # Ends with year
            if int(date_splitted[0]) > 12: # Day is at the beginning
                date_formats = [f'%d{date_sep[0]}%m{date_sep[1]}%Y']
            elif int(date_splitted[1]) > 12: # Day is in the middle
                date_formats = [f'%m{date_sep[0]}%d{date_sep[1]}%Y']
            else: # Not sure if day is at the beginning or in the middle, include both
                date_formats = [
                    f'%d{date_sep[0]}%m{date_sep[1]}%Y',
                    f'%m{date_sep[0]}%d{date_sep[1]}%Y'
                ]
        else:
            raise ValueError(f"{DATETIME_NOT_FOUND_STRING}. Date string with 10 characters: {date_str}")
    
    return date_formats

# %% === Helper method to get compatible time formats
def _get_compatible_time_formats(
        time_str: str
    ) -> list[str]:
    """
    Get a list of compatible time formats from a time string.
    
    Args:
        time_str (str): The time string to get compatible time formats from.
        
    Returns:
        list[str]: A list of compatible time formats.
    """
    time_sep = [x for x in time_str if not x.isdigit()]
    if len(time_sep) == 1: # Found just hour and minute
        time_formats = [f'%H{time_sep[0]}%M']
    elif len(time_sep) == 2: # Found hour, minute and second
        time_formats = [f'%H{time_sep[0]}%M{time_sep[1]}%S']
    else:
        raise ValueError(f"{DATETIME_NOT_FOUND_STRING}. Time string with 0 or more than 2 separators: {time_str}")
    
    return time_formats

# %% Helper function to combine possible date and time formats, to obtain all the possible full datetime formats
def _get_date_time_formats_combination(
        date_formats: list[str], 
        time_formats: list[str]=None,
        date_time_separator: str=None,
        date_first: bool=True
    ) -> set[str]:
    full_formats = set()
    if time_formats is None or len(time_formats) == 0:
        for curr_d_f in date_formats:
            full_formats.add(curr_d_f)
    else:
        for curr_d_f in date_formats:
            for curr_t_f in time_formats:
                if date_first:
                    full_formats.add(f'{curr_d_f}{date_time_separator}{curr_t_f}')
                else:
                    full_formats.add(f'{curr_t_f}{date_time_separator}{curr_d_f}')
    
    return full_formats
        
# %% === Helper method to infer possible datetime formats from a series
def _infer_possible_full_datetime_formats(
        datetime_series: list[str] | np.ndarray | pd.Series |str,
        date_time_separator: str=None
    ) -> set[str]:
    """
    Infer possible datetime formats from a pandas Series.
    
    Args:
        series (pd.Series): The pandas Series to infer datetime formats from.
        date_time_separator (str, optional): The separator between date and time to use (default: None, which means that the existing one will be used).
        
    Returns:
        set[str]: A set of possible datetime formats.
    """
    datetime_series = _validate_datetime_series(datetime_series)

    full_formats = set()
    for full_date_str in datetime_series:
        # if len(full_date_str) not in [8, 10, 16, 19]: # YYYYMMDD, YYYY-MM-DD, YYYY-MM-DD HH:MM, YYYY-MM-DD HH:MM:SS
        #     raise ValueError(date_format_not_found_string)
        
        date_match = DATE_PATTERN.search(full_date_str)
        time_match = TIME_PATTERN.search(full_date_str)

        date_first = True
        if date_match and time_match:
            if date_match.start() > time_match.start():
                date_first = False

            if date_time_separator is None:
                if date_first:
                    date_time_separator = full_date_str[date_match.end():time_match.start()]
                else:
                    date_time_separator = full_date_str[time_match.end():date_match.start()]
        
        if date_match:
            date_str = date_match.group()
            date_formats = _get_compatible_date_formats(date_str)
        else:
            raise ValueError(f"{DATETIME_NOT_FOUND_STRING}. No date match found: {full_date_str}")
        
        time_formats = None
        if time_match:
            time_str = time_match.group()
            time_formats = _get_compatible_time_formats(time_str)
            
        curr_full_formats = _get_date_time_formats_combination(
            date_formats=date_formats, 
            time_formats=time_formats, 
            date_time_separator=date_time_separator, 
            date_first=date_first
        )
        
        full_formats.update(curr_full_formats)

    return full_formats

# %% === Method to parse date with inferred format from infer_datetime_format
def parse_datetime(
        date_str: str, 
        date_format: str, 
        fuzzy: bool=False,
        date_time_separator: str=None
    ) -> pd.Timestamp:
    """
    Parse a date string using the inferred date format.
    
    Args:
        date_str (str): The date string to parse.
        date_format (str): The inferred date format (from infer_datetime_format method).
        fuzzy (bool, optional): Whether to use fuzzy parsing (default: False).
        date_time_separator (str, optional): The separator between date and time (default: None). It has no effect if fuzzy is False or date_format is 'dateutil'.
        
    Returns:
        pd.Timestamp: The parsed date.
    """
    if date_format == 'dateutil':
        return pd.Timestamp(parser.parse(date_str, fuzzy=fuzzy))
    else:
        if fuzzy:
            date_str_clean = _clean_date_string(date_str, date_time_separator)
            return pd.to_datetime(date_str_clean, format=date_format)
        else:
            return pd.to_datetime(date_str, format=date_format)

# %% === Helper function to check and validate datetime_series
def _validate_datetime_series(
    datetime_series: list[str] | np.ndarray | pd.Series |str
    ) -> list[str]:
    if isinstance(datetime_series, str):
        datetime_series = [datetime_series]
    elif isinstance(datetime_series, np.ndarray):
        datetime_series = datetime_series.tolist()
    elif isinstance(datetime_series, pd.Series):
        datetime_series = datetime_series.tolist()
    
    if not isinstance(datetime_series, list):
        raise ValueError("datetime_series must be a list, numpy array, pandas series, or a string.")
    
    return datetime_series.copy()
        
# %% === Helper method to extract single datetime format from multiple compatible formats
def _get_single_datetime_format(
        datetime_series: list[str] | np.ndarray | pd.Series | str,
        formats_detected: list[str],
        fuzzy_allowed: bool=False
    ) -> str:
    datetime_series = _validate_datetime_series(datetime_series)

    # Create compatibility matrix
    compatibility_matrix = np.zeros((len(datetime_series), len(formats_detected)), dtype='bool')
    for i, full_date_str in enumerate(datetime_series):
        for j, date_format in enumerate(formats_detected):
            try:
                parse_datetime(
                    date_str=full_date_str,
                    date_format=date_format,
                    fuzzy=fuzzy_allowed,
                    date_time_separator=None # Ignored, because we just want to test if the method works -> datetime detected
                )
                compatibility_matrix[i, j] = True
            except ValueError:
                pass
    
    compatible_formats = np.all(compatibility_matrix, axis=0)
    compatible_formats_count = np.sum(compatible_formats)
    
    if compatible_formats_count == 1:
        return formats_detected[np.argmax(compatible_formats)] # Return the most compatible format (string)
    elif compatible_formats_count > 1:
        compatible_formats_list = [formats_detected[i] for i, x in enumerate(compatible_formats) if x]
        raise ValueError(f"Multiple compatible date formats detected for the entire series: {compatible_formats_list}")
    else:
        raise ValueError("No compatible date format found over the entire series")

# %% === Method to infer date format
def infer_datetime_format(
        datetime_strings: list[str] | np.ndarray | pd.Series |str, 
        allow_multiple_formats: bool=False,
        samples_to_test: int=50,
        fuzzy: bool=False,
        date_time_separator: str=None
    ) -> str:
    """
    Infer the date format of a pandas Series.
    
    Args:
        series (pd.Series): The pandas Series to infer the date format from.
        allow_multiple_formats (bool, optional): Whether to allow multiple date formats (default: False).
        samples_to_test (int, optional): The number of samples to test (default: 50).
        fuzzy (bool, optional): Whether to use fuzzy parsing (default: False).
        date_time_separator (str, optional): The separator between date and time to use internally (default: None).
        
    Returns:
        str: The inferred date format (if 'dateutil' is returned, it means that multiple formats are allowed and dateutil module can be used).
    """
    datetime_series = _validate_datetime_series(datetime_strings)
    datetime_series_head = datetime_series[:samples_to_test]

    try:
        if allow_multiple_formats:
            # Try to parse dates using dateutil parser
            try:
                [parser.parse(x) for x in datetime_series_head] # Test if parser works and do not return errors on first elements of datetime_strings
                return 'dateutil'
            except Exception as e:
                raise ValueError(f"{DATETIME_NOT_FOUND_STRING}. Multiple formats are allowed, but dateutil parser failed: {e}")
        else:
            formats = _infer_possible_full_datetime_formats(
                datetime_series=datetime_series_head, 
                date_time_separator=date_time_separator
            )
            
            if len(formats) == 1:
                return list(formats)[0]
            else:
                return _get_single_datetime_format(
                    datetime_series=datetime_series_head, 
                    formats_detected=list(formats), 
                    fuzzy_allowed=fuzzy
                )
    
    except Exception as e:
        raise ValueError(f"Error inferring date format: {str(e)}")

# %%