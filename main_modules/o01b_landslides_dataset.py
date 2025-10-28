# %% === Import necessary modules
import os
import sys
import argparse
import warnings
import pandas as pd
import geopandas as gpd
import shapely.geometry as geom

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    REFERENCE_POINTS_FILENAME,
    SUPPORTED_FILE_TYPES
)

# Importing necessary modules from psliptools
from psliptools.geometries import (
    load_vectorial_file_geometry,
    get_geo_file_fields,
    create_polygons_from_points,
    convert_polygons_crs
)

from psliptools.utilities import (
    select_file_prompt,
    select_from_list_prompt,
    read_generic_csv
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
from main_modules.m01a_study_area import get_proj_epsg_and_add_prj_coords_to_df
logger = setup_logger(__name__)
logger.info("=== Create landslides dataset ===")

# %% === Helper functions and global variables
INTERNAL_EPSG_CODE = 4326 # Remember that if you modify this, you should also check and possibly modify the rest of the code!
REQUIRED_LANDSLIDES_DF_COLUMNS = ['lon', 'lat', 'id', 'src_file']
DEFAULT_VECTORIAL_EXTENSION = '.shp'

def read_landslides_csv(
        landslide_points_csv_path: str
    ) -> pd.DataFrame:
    """Reads a landslide points csv file and returns a pandas DataFrame."""
    landslides_df = read_generic_csv(csv_path=landslide_points_csv_path)

    if landslides_df.empty:
        raise ValueError(f"Reference points CSV ({landslide_points_csv_path}) is empty. Please check the file.")
    
    if not all(x in landslides_df.columns for x in REQUIRED_LANDSLIDES_DF_COLUMNS):
        raise ValueError(f"Reference points CSV ({landslide_points_csv_path}) does not contain the required columns: {REQUIRED_LANDSLIDES_DF_COLUMNS}. Please check the file.")
    
    return landslides_df

def load_vectorials_w_mapper(
        file_mapper: dict[str, list[str, str]]
    ) -> dict[str, pd.DataFrame]:
    """Loads the vectorials and returns a dictionary with their geometries."""
    source_geoms_geo = {}
    for src_shp, (src_shp_path, src_shp_id_field) in file_mapper.items():
        curr_df = load_vectorial_file_geometry( # TODO: Speed up this function
            file_path=src_shp_path,
            field=src_shp_id_field,
            convert_to_geo=True,
            allow_only_polygons=False
        )

        source_geoms_geo[src_shp] = curr_df
    
    return source_geoms_geo

def create_landslides_polygons(
        landslides_df: pd.DataFrame,
        source_geometries_geo: dict[str, pd.DataFrame],
        file_mapper: dict[str, list[str, str]],
        point_buffer: float,
        point_buffer_type: str,
        proj_epsg: int
    ) -> gpd.GeoDataFrame:
    """Creates the landslide polygons from the reference points."""
    landslides_df['area_in_sqm'] = None
    landslides_df['fake_poly'] = None
    landslides_df['geometry'] = None
    for idx, row in landslides_df.iterrows():
        src_file = str(row['src_file'])
        create_fake_polygon = (src_file is None) or (src_file == "") or (src_file == "nan")

        if not create_fake_polygon:
            curr_id = row['id']
            idx_in_src_shp = source_geometries_geo[src_file][source_geometries_geo[src_file]['class_name'] == curr_id].index
            if idx_in_src_shp.empty:
                warnings.warn(f"ID {curr_id} not found in file ({file_mapper[src_file][0]}). Fake polygon will be created but please, check the reference points CSV.")
                create_fake_polygon = True
            elif len(idx_in_src_shp) > 1:
                raise ValueError(f"Multiple IDs ({curr_id}) found in file ({file_mapper[src_file][0]}). Please check the reference points CSV.")
        
        if create_fake_polygon:
            curr_proj_x = row['prj_x']
            curr_proj_y = row['prj_y']
            curr_proj_poly = create_polygons_from_points(x=curr_proj_x, y=curr_proj_y, buffer=point_buffer, shape=point_buffer_type)[0]
            curr_geo_poly = convert_polygons_crs(polygons=curr_proj_poly, crs_in=proj_epsg, crs_out=INTERNAL_EPSG_CODE)[0]
        else:
            curr_geo_poly = source_geometries_geo[src_file].loc[idx_in_src_shp[0], 'geometry']
            if not isinstance(curr_geo_poly, (geom.Polygon, geom.MultiPolygon)):
                create_fake_polygon = True
                curr_geo_poly = curr_geo_poly.buffer(1e-8) # Add a small buffer
                curr_proj_poly = convert_polygons_crs(polygons=curr_geo_poly, crs_in=INTERNAL_EPSG_CODE, crs_out=proj_epsg)[0].buffer(point_buffer)
                curr_geo_poly = convert_polygons_crs(polygons=curr_proj_poly, crs_in=proj_epsg, crs_out=INTERNAL_EPSG_CODE)[0]
            else:
                curr_proj_poly = convert_polygons_crs(polygons=curr_geo_poly, crs_in=INTERNAL_EPSG_CODE, crs_out=proj_epsg)[0]
        
        landslides_df.loc[idx, 'area_in_sqm'] = curr_proj_poly.area
        landslides_df.loc[idx, 'fake_poly'] = create_fake_polygon
        landslides_df.loc[idx, 'geometry'] = curr_geo_poly
    
    landslides_gdf = gpd.GeoDataFrame(landslides_df, geometry='geometry', crs=f"EPSG:{INTERNAL_EPSG_CODE}")

    return landslides_gdf

def write_vectorial_file(
        landslides_gdf: gpd.GeoDataFrame,
        out_dir: str,
        out_filename: str=None,
        out_epsg_code: int=INTERNAL_EPSG_CODE
    ) -> None:
    if out_filename is None:
        out_filename = f"pslip_landslides_epsg_{out_epsg_code}{DEFAULT_VECTORIAL_EXTENSION}"
    if not any([out_filename.endswith(x) for x in SUPPORTED_FILE_TYPES['vectorial']]):
        out_filename += DEFAULT_VECTORIAL_EXTENSION
    
    out_filepath = out_filename
    if not out_filepath.startswith(out_dir):
        out_filename = os.path.basename(out_filename)
        out_filepath = os.path.join(out_dir, out_filename)
    
    landslides_gdf.to_crs(epsg=out_epsg_code, inplace=False).to_file(out_filepath)

# %% === Main function to define the study area
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        points_csv_path: str=None,
        point_buffer: float=10, # in meters
        point_buffer_type: str='circle', # circle or square
        file_mapper: dict[str, list[str, str]]=None,
        write_out_file: bool=True,
        out_filename: str=None,
        out_epsg_code: int=INTERNAL_EPSG_CODE
    ) -> dict[str, object]:
    """Main function to define the study area."""
    src_type = 'landslides'

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=True)

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Reference points file selection ===")
        if points_csv_path is None:
            points_csv_path = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the reference points csv (Default: {REFERENCE_POINTS_FILENAME}): ",
                src_ext='csv',
                default_file=os.path.join(env.folders['user_control']['path'], REFERENCE_POINTS_FILENAME)
            )
    
    logger.info(f"Reading reference points CSV: {points_csv_path}")
    landslides_df = read_landslides_csv(points_csv_path)
    
    logger.info("Converting reference points coordinates into projected...")
    lands_proj_epsg, landslides_df = get_proj_epsg_and_add_prj_coords_to_df(dataframe=landslides_df)
    
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        files_list = list(set(landslides_df['src_file'][~landslides_df['src_file'].isna()].to_list()))
        
        if file_mapper is None:
            file_mapper = {x: [None, None] for x in files_list}
        
        if not all([x in file_mapper.keys() for x in files_list]):
            raise ValueError("Some of the shapefiles in the reference points CSV are not in the file_mapper. Please check the file or update the file_mapper.")
        
        if any([x[0] is None or x[1] is None for x in file_mapper.values()]):
            print("\n=== File mapper association ===")
            for src_shp in file_mapper.keys():
                if file_mapper[src_shp][0] is None:
                    src_ext = os.path.splitext(src_shp)[1]
                    if src_ext == '':
                        src_ext = DEFAULT_VECTORIAL_EXTENSION
                    elif src_ext not in SUPPORTED_FILE_TYPES['vectorial']:
                        raise ValueError(f"Invalid extension for shapefile [{src_shp}]. Must be one of: {SUPPORTED_FILE_TYPES['vectorial']}.")
                    
                    src_shp_w_ext = f"{src_shp}{src_ext}" if not src_shp.endswith(src_ext) else src_shp
                    file_mapper[src_shp][0] = select_file_prompt(
                        base_dir=env.folders['inputs'][src_type]['path'],
                        usr_prompt=f"Name or full path of the {src_shp} shapefile (default: {os.path.join(env.folders['inputs'][src_type]['path'], src_shp_w_ext)}): ",
                        src_ext=src_ext,
                        default_file=src_shp_w_ext
                    )
                if file_mapper[src_shp][1] is None:
                    shp_fields, shp_types = get_geo_file_fields(file_mapper[src_shp][0])
                    file_mapper[src_shp][1] = select_from_list_prompt(
                        obj_list=shp_fields, 
                        obj_type=shp_types, 
                        usr_prompt=f"Select the field containing the id for shapefile {file_mapper[src_shp][0]}:", 
                        allow_multiple=False
                    )[0]
    
    logger.info("Loading source shapefiles...")
    source_shapefiles_geo = load_vectorials_w_mapper(file_mapper)

    logger.info("Creating landslide polygons...")
    landslides_gdf = create_landslides_polygons(
        landslides_df=landslides_df,
        source_geometries_geo=source_shapefiles_geo,
        file_mapper=file_mapper,
        point_buffer=point_buffer,
        point_buffer_type=point_buffer_type,
        proj_epsg=lands_proj_epsg
    )

    if write_out_file:
        logger.info("Writing landslide polygons to file...")
        write_vectorial_file(
            landslides_gdf=landslides_gdf,
            out_dir=env.folders['inputs'][src_type]['path'],
            out_filename=out_filename,
            out_epsg_code=out_epsg_code
        )

    landslides_vars = {
        'geodataframe': landslides_gdf,
        'current_epsg': INTERNAL_EPSG_CODE,
        'proj_epsg': lands_proj_epsg,
        'point_buffer': point_buffer,
        'point_buffer_type': point_buffer_type
    }

    env.save_variable(variable_to_save=landslides_vars, variable_filename='landslides_vars.pkl')
    
    return landslides_vars

 # %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the study area for the analysis.")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for the analysis.")
    parser.add_argument('--gui_mode', action='store_true', help="Run in GUI mode (not implemented yet).")
    parser.add_argument('--points_csv_path', type=str, default=None, help="Path to the reference points csv file.")
    parser.add_argument('--point_buffer', type=float, default=10, help="Point buffer in meters.")
    parser.add_argument('--point_buffer_type', type=str, default="circle", help="Point buffer type (circle, square).")
    parser.add_argument('--file_mapper', type=str, nargs='+', help="Shapefile mapper (shapefile, id_field).")
    parser.add_argument('--write_out_file', action='store_true', help="Write out the shapefile.")
    parser.add_argument('--out_filename', type=str, default=None, help="Output shapefile filename.")
    parser.add_argument('--out_epsg_code', type=int, default=INTERNAL_EPSG_CODE, help="Output shapefile EPSG code.")
    args = parser.parse_args()

    study_area_vars = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode,
        landslides_points_csv_path=args.points_csv_path,
        point_buffer=args.point_buffer,
        point_buffer_type=args.point_buffer_type,
        file_mapper=args.file_mapper,
        write_out_file=args.write_out_file,
        out_filename=args.out_filename,
        out_epsg_code=args.out_epsg_code
    )

# %%
