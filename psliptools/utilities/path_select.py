#%% Import necessary modules
import os

#%% file_selector
def file_selector(base_dir: str, src_ext: str=None) -> list[str]:
    if not(os.path.isdir(base_dir)):
        raise ValueError("base_dir must be a valid directory")
    
    print("Available files:")
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
    
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")

    files_sel = [
        x.strip(' "') for x in input(
            "Enter the number of the file you want to select" \
            +" (also multiple, comma or semicolon separated): "
        ).replace(',', ';').split(';')
    ]
    if not files_sel:
        files_sel = [str(i) for i in range(1, len(files)+1)]

    files_sel = sorted(set(
        files[int(x)-1] 
        if x.isdigit() and 1 <= int(x) <= len(files) 
        else x
        for x in files_sel
    ))

    full_files_sel = [os.path.join(base_dir, f) for f in files_sel]
    return full_files_sel

# %%
