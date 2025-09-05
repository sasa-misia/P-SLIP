# %% === Import necessary modules
import os
import warnings

# %% === Function to check if an object is a list containing strings
def _check_list_of_strings(obj_list) -> None:
    """Check if an object is a list containing strings."""
    if obj_list is None:
        raise ValueError("obj_list must not be empty")
    if not isinstance(obj_list, list):
        raise ValueError(f"obj_list must be a list, not {type(obj_list)}")
    for i, obj in enumerate(obj_list):
        if not isinstance(obj, str):
            raise ValueError(f"obj_list must be a list of strings, but found {type(obj)} at index {i}")
        
# %% === Function to parse the files in a folder
def _parse_files_in_folder(
        base_dir: str,
        src_ext: str=None
    ) -> list[str]:
    """
    Parse the files in a folder.

    Args:
        base_dir (str): The base directory to parse.
        src_ext (str, optional): The file extension to filter by. Defaults to None.

    Returns:
        list[str]: A list of the files contained in the folder and possibly filtered by extension.
    """
    if not os.path.isdir(base_dir):
        raise ValueError("base_dir must be a valid directory")
    
    if src_ext is None:
        files = [
            f for f in os.listdir(base_dir) 
            if os.path.isfile(os.path.join(base_dir, f))
        ]
    elif isinstance(src_ext, str):
        files = [
            f for f in os.listdir(base_dir) 
            if os.path.isfile(os.path.join(base_dir, f)) and 
            f.endswith(src_ext)
        ]
    elif isinstance(src_ext, list):
        files = [
            f for f in os.listdir(base_dir) 
            if os.path.isfile(os.path.join(base_dir, f)) and 
            any(
                f.endswith(ext) for ext in src_ext
            )
        ]
    else:
        raise ValueError("src_ext must be None, a string or a list of strings")
    return files

# %% === Function to parse a string with a selection
def _parse_selection_string(
        possible_values: list[str],
        raw_string: str=None
    ) -> list[int]:
    """
    Parse a string with a selection of values.

    Args:
        possible_values (list[str]): A list of the possible values.
        raw_string (str, optional): The raw string to parse. Defaults to None (all values are selected).

    Returns:
        list[int]: A list of selected indices.
    """
    _check_list_of_strings(possible_values)
    raw_string = raw_string.strip(' "')

    if raw_string:
        # selected_indices = [
        #     int(x.strip(' "')) 
        #     if x.isdigit()
        #     else possible_values.index(x) + 1
        #     for x in raw_string.replace(';', ',').split(',')
        # ]
        selected_indices = []
        for i, sel in enumerate(raw_string.replace(';', ',').split(',')):
            sel = sel.strip(' "')
            if sel.isdigit():
                selected_indices.append(int(sel))
            elif sel in possible_values:
                selected_indices.append(possible_values.index(sel) + 1)
            elif len(sel.split(':')) == 2 and sel.split(':')[0].isdigit() and sel.split(':')[1].isdigit():
                selected_indices.extend(range(
                    int(sel.split(':')[0]), 
                    int(sel.split(':')[1]) + 1
                ))
            else:
                raise ValueError(f"Invalid input at index {i}: {sel}")
    else:
        selected_indices = [i for i in range(1, len(possible_values) + 1)]
        
    for i, x in enumerate(selected_indices):
        if not isinstance(x, int):
            raise ValueError(f"Element at index {i} of selected_indices is not an integer ({x} is {type(x)})")
        if x < 1 or x > len(possible_values):
            raise ValueError(f"Element at index {i} of selected_indices is out of range ({x} is out of 1:{len(possible_values)})")
    return selected_indices

# %% === Function to print an enumerate list of strings
def print_enumerated_list(
        obj_list: list[str],
        obj_type: list[str]=None
    ) -> None:
    """
    Print an enumerate list of strings.

    Args:
        obj_list (list[str]): List of strings to enumerate.
        obj_type (list[str], optional): List of types for each object in obj_list.
    
    Returns:
        None
    """
    _check_list_of_strings(obj_list)

    if not obj_type:
        obj_type = [""] * len(obj_list)
    else:
        obj_type = ['[type: ' + x + ']' for x in obj_type]

    _check_list_of_strings(obj_type)

    if len(obj_list) != len(obj_type):
        raise ValueError("obj_list and obj_type must have the same length")
    
    print("Available options:")
    for i, (o, t) in enumerate(zip(obj_list, obj_type)):
        print(f"{i+1}| {o} {t}")

# %% === Function to select one or multiple options from a list
def select_from_list_prompt(
        obj_list: list[str],
        usr_prompt: str=None,
        allow_multiple: bool = False,
        obj_type: list[str]=None
    ) -> list[str]:
    """
    Select one or more options from a list.

    Args:
        obj_list (list[str]): List of options to select from.
        allow_multiple (bool, optional): If True, allows multiple selections.

    Returns:
        list[int]: List of selected indices.
    """
    _check_list_of_strings(obj_list)

    if usr_prompt:
        if not isinstance(usr_prompt, str):
            raise ValueError(f"usr_prompt must be a string, not {type(usr_prompt)}")
        if not usr_prompt.endswith(": "):
            usr_prompt += ": "
    else:
        usr_prompt = 'Enter your selection: '
        
    if allow_multiple:
        default_index = f"1:{len(obj_list)}"
        multi_completion = "(s)"
        extra_prompt = "\n    - Multiple selection -> use [,] or [;] (ex. 1, 2, 3) \n    - Range selection -> use [:] (ex. 1:3)"
    else:
        default_index = "1"
        multi_completion = ""
        extra_prompt = ""
    
    print_enumerated_list(obj_list, obj_type)

    print(f"\n You can specify the number{multi_completion} or name{multi_completion} " \
        + f"of the option{multi_completion} you want to select (or press enter for default: {default_index}) " \
        + f"{extra_prompt}")

    user_raw_input = input("\n"+usr_prompt).strip(' "')

    selected_indices = _parse_selection_string(obj_list, user_raw_input)
    
    if not allow_multiple and len(selected_indices) > 1:
        warnings.warn("Multiple selections are not allowed when allow_multiple is False, first element will be used", UserWarning)
        selected_indices = [selected_indices[0]]
    
    selected_objs = sorted(set(
        obj_list[int(x)-1]
        for x in selected_indices
    )) # Sort and remove duplicates
    return selected_objs

# %% === Function to select one directory
def select_dir_prompt(
        default_dir: str = None,
        content_type: str = None
    ) -> str:
    """
    Select a directory from a base directory.

    Args:
        default_dir (str, optional): Default directory to select.
        content_type (str, optional): Type of content in the directory.

    Returns:
        str: Selected directory.
    """
    if default_dir is None:
        default_dir = os.getcwd()
    if content_type is None:
        content_type = "your"

    selected_fold = input(
        f"\n Enter the folder name where {content_type} files are stored (default: {default_dir}): "
    ).strip(' "')
    if not selected_fold:
        selected_fold = default_dir
    
    if not os.path.isdir(selected_fold):
        raise ValueError(f"{selected_fold} is not a valid directory")
    return selected_fold

# %% === Function to select one or multiple files in a folder
def select_files_in_folder_prompt(
        base_dir: str=None,
        usr_prompt: str=None,
        allow_multiple: bool = False,
        src_ext: str | list[str]=None
    ) -> list[str]:
    """
    Select files from a directory.

    Args:
        base_dir (str): Base directory to select files from.
        src_ext (str, optional): File extension to filter by.

    Returns:
        list[str]: List of selected files.
    """
    if base_dir is None:
        base_dir = select_dir_prompt()
    if not(os.path.isdir(base_dir)):
        raise ValueError("base_dir must be a valid directory")
    
    files = _parse_files_in_folder(base_dir, src_ext)
    
    files_sel = select_from_list_prompt(files, usr_prompt, allow_multiple)

    full_files_sel = [os.path.join(base_dir, f) for f in files_sel]

    for f in full_files_sel:
        if not os.path.isfile(f):
            raise ValueError(f"{f} is not a valid file")
    return full_files_sel

# %% === Function to select a single file
def select_file_prompt(
        base_dir: str=None,
        usr_prompt: str=None,
        src_ext: str | list[str]=None,
        default_file: str=None
    ) -> str:
    if not base_dir:
        base_dir = select_dir_prompt()

    if not usr_prompt:
        usr_prompt = 'Enter the name of the file or full path'
        if default_file:
            usr_prompt += f" (default: [{default_file}])"
        usr_prompt += ": "

    files = _parse_files_in_folder(base_dir, src_ext=src_ext)

    if files:
        print_enumerated_list(files)

    source_path = input("\n"+usr_prompt).strip(' "')
    if not source_path and default_file:
        source_path = default_file

    if files and source_path.isdigit() and int(source_path) in range(1, len(files)+1):
        source_path = files[int(source_path)-1]
    
    if not os.path.isabs(source_path):
        source_path = os.path.join(base_dir, source_path)
    
    if not os.path.isfile(source_path):
        raise ValueError(f"{source_path} is not a valid file")
    return source_path

# %% ===
