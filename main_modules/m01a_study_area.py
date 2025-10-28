# %% === Import necessary modules
import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    REFERENCE_POINTS_FILENAME,
    SUPPORTED_FILE_TYPES
)

# Importing necessary modules from psliptools
from psliptools.geometries import (
    load_vectorial_file_geometry,
    get_rectangle_parameters,
    create_rectangle_polygons,
    union_polygons,
    get_polygon_extremes,
    get_geo_file_fields,
    get_geo_file_field_attributes,
    create_polygons_from_points,
    convert_polygons_crs
)

from psliptools.utilities import (
    select_file_prompt,
    select_from_list_prompt,
    read_generic_csv
)

from psliptools.rasters import (
    get_projected_epsg_code_from_bbox,
    convert_coords,
    get_unit_of_measure_from_epsg
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger(__name__)
logger.info("=== Import or Create Study Area ===")

# %% === Helper functions and global variables
REM_POLY_DF = pd.DataFrame(columns=['type', 'subtype', 'class_name', 'geometry']) # Empty DataFrame for removed areas
SOURCE_MODE_ALLOWED = ['user_bbox', 'geo_file', 'reference_points']

def define_study_area_from_user_bbox(
        rectangle_polygons: list[object]
    ) -> dict[str, object]:
    """Define study area from rectangle polygons."""
    rect_polys = rectangle_polygons.copy()
    rect_names = [f"Poly {i+1}" for i in range(len(rectangle_polygons))]
    study_area_df = pd.DataFrame({
        'class_name': rect_names,
        'geometry': rect_polys
    })
    study_area_poly = union_polygons(rect_polys)
    study_area_extremes = get_polygon_extremes(study_area_poly)
    study_area_vars = {
        'study_area_polygon': study_area_poly,
        'study_area_cls_poly': study_area_df,
        'study_area_cln_poly': study_area_poly,
        'study_area_rem_poly': REM_POLY_DF,
        'study_area_extremes': study_area_extremes
    }

    return study_area_vars

def define_study_area_from_geo_file(
        file_path: str, 
        field: str = None, 
        attributes: list[str] = None
    ) -> dict[str, object]:
    """Define study area from a shapefile (or geopackage, etc), optionally clipping with custom polygons."""
    study_area_df = load_vectorial_file_geometry(
        file_path=file_path, 
        field=field, 
        attributes=attributes,
        convert_to_geo=True,
        allow_only_polygons=True
    )
    polygons = study_area_df['geometry']
    study_area_poly = union_polygons(polygons)
    study_area_extremes = get_polygon_extremes(study_area_poly)
    study_area_vars = {
        'study_area_polygon': study_area_poly,
        'study_area_cls_poly': study_area_df,
        'study_area_cln_poly': study_area_poly,
        'study_area_rem_poly': REM_POLY_DF,
        'study_area_extremes': study_area_extremes
    }

    return study_area_vars

def get_proj_epsg_and_add_prj_coords_to_df(
        dataframe: pd.DataFrame,
        x_column: str='lon',
        y_column: str='lat',
        input_crs: int=4326
    ) -> tuple[int, pd.DataFrame]:
    """
    Get the most suitable projection EPSG code and add the projected coordinates to the dataframe (obtained from lon and lat columns).
    
    Args:
        dataframe (pd.DataFrame): The dataframe containing the longitude and latitude columns.
        
    Returns:
        tuple[int, pd.DataFrame]: A tuple containing the EPSG code and the dataframe with the projected coordinates.
    """
    dataframe = dataframe.copy()
    landslides_bbox = (dataframe[x_column].min(), dataframe[y_column].min(), dataframe[x_column].max(), dataframe[y_column].max())

    proj_epsg = get_projected_epsg_code_from_bbox(geo_bbox=landslides_bbox)

    if get_unit_of_measure_from_epsg(proj_epsg) != "meter":
        raise ValueError("The projection EPSG code must be in meters.")

    dataframe['prj_x'], dataframe['prj_y'] = convert_coords(
        crs_in=input_crs, # Because lon and lat were given
        crs_out=proj_epsg,
        in_coords_x=dataframe[x_column],
        in_coords_y=dataframe[y_column]
    )

    return proj_epsg, dataframe

def define_study_area_from_reference_points(
        file_path: str, 
        point_buffer: float, # In meters
        point_buffer_type: str # circle or square
    ) -> dict[str, object]:
    """Define study area from reference points."""
    ref_df = read_generic_csv(file_path)

    proj_epsg, ref_df = get_proj_epsg_and_add_prj_coords_to_df(ref_df)

    ref_df['proj_buffer_poly'] = None
    ref_df['geo_buffer_poly'] = None
    for idx, row in ref_df.iterrows():
        curr_proj_poly = create_polygons_from_points(x=row['prj_x'], y=row['prj_y'], buffer=point_buffer, shape=point_buffer_type)[0]
        curr_geo_poly = convert_polygons_crs(polygons=curr_proj_poly, crs_in=proj_epsg, crs_out=4326)[0]
        ref_df.loc[idx, 'proj_buffer_poly'] = curr_proj_poly
        ref_df.loc[idx, 'geo_buffer_poly'] = curr_geo_poly

    study_area_df = pd.DataFrame({
        'class_name': ref_df['id'],
        'geometry': ref_df['geo_buffer_poly'],
        'buffer_type': point_buffer_type,
        'buffer_in_meters': point_buffer
    })

    study_area_poly = union_polygons(ref_df['geo_buffer_poly'])
    study_area_extremes = get_polygon_extremes(study_area_poly)
    study_area_vars = {
        'study_area_polygon': study_area_poly,
        'study_area_cls_poly': study_area_df,
        'study_area_cln_poly': study_area_poly,
        'study_area_rem_poly': REM_POLY_DF,
        'study_area_extremes': study_area_extremes
    }

    return study_area_vars

# %% === Main function to define the study area
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        source_mode: str="geo_file",
        point_buffer: float=2000, # in meters
        point_buffer_type: str="square",
    ) -> dict[str, object]:
    """Main function to define the study area."""
    if not source_mode in SOURCE_MODE_ALLOWED:
        raise ValueError("Invalid source mode. Must be one of: " + ", ".join(SOURCE_MODE_ALLOWED))
    
    source_type = 'study_area'

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=True)

    # Initialize variables
    source_path, source_field, source_attributes = None, None, None

    # --- User choices section ---
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        if source_mode == 'user_bbox': # === User bounding box ===
            n_rectangles = int(input("How many bounding boxes? [1]: ") or "1")
            rec_polys = create_rectangle_polygons(get_rectangle_parameters(n_rectangles))
        elif source_mode == 'geo_file': # === Geo file ===
            print("\n=== Vectorial file selection ===")
            source_path = select_file_prompt(
                base_dir=env.folders['inputs'][source_type]['path'],
                usr_prompt=f"Name or full path of the {source_type} vectorial file (ex. {source_type}.shp): ",
                src_ext=SUPPORTED_FILE_TYPES['vectorial']
            )

            shp_fields, shp_types = get_geo_file_fields(source_path)
            print("\n=== Vectorial file field ===")
            source_field = select_from_list_prompt(
                obj_list=shp_fields, 
                obj_type=shp_types, 
                usr_prompt="Select the field:", 
                allow_multiple=False
            )[0]

            shp_field_vals = get_geo_file_field_attributes(source_path, source_field, sort=True)
            print("\n=== Vectorial file attributes ===")
            source_attributes = select_from_list_prompt(
                obj_list=shp_field_vals,
                usr_prompt=f"Select the attribute(s) inside the field ({source_field}):", 
                allow_multiple=True
            )
        elif source_mode == 'reference_points': # === Reference points ===
            print("\n=== Reference points file selection ===")
            source_path = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the reference points csv (Default: {REFERENCE_POINTS_FILENAME}): ",
                src_ext=SUPPORTED_FILE_TYPES['table']
            )
        else:
            raise ValueError(f"Invalid source mode: {source_mode}")
    
    if source_mode == 'user_bbox':
        study_area_vars = define_study_area_from_user_bbox(rec_polys)
    elif source_mode == 'geo_file':
        study_area_vars = define_study_area_from_geo_file(
            file_path=source_path, 
            field=source_field, 
            attributes=source_attributes
        )
    elif source_mode == 'reference_points':
        study_area_vars = define_study_area_from_reference_points(
            file_path=source_path, 
            point_buffer=point_buffer, 
            point_buffer_type=point_buffer_type
        )
    else:
        raise ValueError(f"Invalid source mode: {source_mode}")

    env.config['inputs'][source_type][0]['settings'] = {
        'source_mode': source_mode,
        'source_field': source_field,
        'source_attributes': source_attributes,
        'source_refined': False
    }

    if source_path:
        _, cust_id = env.add_input_file(file_path=source_path, file_type=source_type)
        env.config['inputs'][source_type][0]['custom_id'] = [cust_id]
    
    env.collect_input_files(file_type=[source_type], multi_extension=True)

    env.save_variable(variable_to_save=study_area_vars, variable_filename=f"{source_type}_vars.pkl")
    
    return study_area_vars

 # %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the study area for the analysis.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    parser.add_argument('--source_mode', type=str, default="geo_file", help="Source mode (user_bbox, geo_file, reference_points).")
    parser.add_argument('--point_buffer', type=float, default=2000, help="Point buffer in meters.")
    parser.add_argument('--point_buffer_type', type=str, default="circle", help="Point buffer type (circle, square).")
    args = parser.parse_args()

    study_area_vars = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode,
        source_mode=args.source_mode,
        point_buffer=args.point_buffer,
        point_buffer_type=args.point_buffer_type
    )

# %%
