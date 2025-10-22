# %% === Import necessary modules
import os
import sys
import argparse
import chardet
import warnings
import pandas as pd
import geopandas as gpd
import shapely.geometry as geom

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    REFERENCE_POINTS_FILENAME
)

# Importing necessary modules from psliptools
from psliptools.geometries import (
    load_shapefile_geometry,
    get_shapefile_fields,
    create_polygons_from_points,
    convert_polygons_crs
)

from psliptools.utilities import (
    select_file_prompt,
    select_from_list_prompt
)

from psliptools.rasters import (
    get_projected_epsg_code_from_bbox,
    convert_coords
)

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger(__name__)
logger.info("=== Create landslides dataset ===")

# %% === Helper functions and global variables
INTERNAL_EPSG_CODE = 4326 # Remember that if you modify this, you should also check and possibly modify the rest of the code!
REQUIRED_LANDSLIDES_DF_COLUMNS = ['lon', 'lat', 'id', 'shapefile']

# TODO: Create helper functions from main()
# TODO: Add logger info strings inside main()

# %% === Main function to define the study area
def main(
        base_dir: str=None,
        gui_mode: bool=False,
        landslide_points_csv_path: str=None,
        point_buffer: float=10, # in meters
        point_buffer_type: str='circle',
        shapefile_mapper: dict[str, list[str, str]]=None,
        write_out_shapefile: bool=True,
        out_shapefile_epsg_code: int=INTERNAL_EPSG_CODE,
        out_shapefile_filename: str=None
    ) -> dict[str, object]:
    """Main function to define the study area."""
    src_type = 'landslides'

    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=True)

    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        print("\n=== Reference points file selection ===")
        if landslide_points_csv_path is None:
            landslide_points_csv_path = select_file_prompt(
                base_dir=env.folders['user_control']['path'],
                usr_prompt=f"Name or full path of the reference points csv (Default: {REFERENCE_POINTS_FILENAME}): ",
                src_ext='csv',
                default_file=os.path.join(env.folders['user_control']['path'], REFERENCE_POINTS_FILENAME)
            )
    
    with open(landslide_points_csv_path, 'rb') as f:
        result = chardet.detect(f.read())
        charenc = result['encoding']

    landslides_df = pd.read_csv(landslide_points_csv_path, encoding=charenc, sep=None, engine='python')
    if landslides_df.empty:
        raise ValueError(f"Reference points CSV ({landslide_points_csv_path}) is empty. Please check the file.")
    
    if not all(x in landslides_df.columns for x in REQUIRED_LANDSLIDES_DF_COLUMNS):
        raise ValueError(f"Reference points CSV ({landslide_points_csv_path}) does not contain the required columns: {REQUIRED_LANDSLIDES_DF_COLUMNS}. Please check the file.")
    
    landslides_bbox = (landslides_df['lon'].min(), landslides_df['lat'].min(), landslides_df['lon'].max(), landslides_df['lat'].max())
    proj_epsg_landslides = get_projected_epsg_code_from_bbox(geo_bbox=landslides_bbox)
    landslides_df['proj_x'], landslides_df['proj_y'] = convert_coords(
        crs_in=INTERNAL_EPSG_CODE,
        crs_out=proj_epsg_landslides,
        in_coords_x=landslides_df['lon'],
        in_coords_y=landslides_df['lat']
    )
    
    if gui_mode:
        raise NotImplementedError("GUI mode is not supported in this script yet. Please run the script without GUI mode.")
    else:
        shapefiles_list = list(set(landslides_df['shapefile'][~landslides_df['shapefile'].isna()].to_list()))
        
        if shapefile_mapper is None:
            shapefile_mapper = {x: [None, None] for x in shapefiles_list}
        
        if not all([x in shapefile_mapper.keys() for x in shapefiles_list]):
            raise ValueError("Some of the shapefiles in the reference points CSV are not in the source_shapefile_mapper. Please check the file or update the source_shapefile_mapper.")
        
        if any([x[0] is None or x[1] is None for x in shapefile_mapper.values()]):
            print("\n=== Shapefile mapper association ===")
            for src_shp in shapefile_mapper.keys():
                if shapefile_mapper[src_shp][0] is None:
                    src_shp_w_ext = f"{src_shp}.shp" if not src_shp.endswith('.shp') else src_shp
                    shapefile_mapper[src_shp][0] = select_file_prompt(
                        base_dir=env.folders['inputs'][src_type]['path'],
                        usr_prompt=f"Name or full path of the {src_shp} shapefile (default: {os.path.join(env.folders['inputs'][src_type]['path'], src_shp_w_ext)}): ",
                        src_ext='shp',
                        default_file=src_shp_w_ext
                    )
                if shapefile_mapper[src_shp][1] is None:
                    shp_fields, shp_types = get_shapefile_fields(shapefile_mapper[src_shp][0])
                    shapefile_mapper[src_shp][1] = select_from_list_prompt(
                        obj_list=shp_fields, 
                        obj_type=shp_types, 
                        usr_prompt=f"Select the field containing the id for shapefile {shapefile_mapper[src_shp][0]}:", 
                        allow_multiple=False
                    )[0]
    

    # TODO: Speed up from this point
    source_shapefiles_geo = {}
    for src_shp, (src_shp_path, src_shp_id_field) in shapefile_mapper.items():
        curr_df = load_shapefile_geometry(
            shapefile_path=src_shp_path,
            field_name=src_shp_id_field,
            convert_to_geo=True,
            allow_only_polygons=False
        )
        source_shapefiles_geo[src_shp] = curr_df

    landslides_df['geometry'] = None
    landslides_df['area_in_sq_meters'] = None
    landslides_df['generated_from_point'] = None
    for idx, row in landslides_df.iterrows():
        src_shp = str(row['shapefile'])
        create_fake_polygon = (src_shp is None) or (src_shp == "") or (src_shp == "nan")

        if not create_fake_polygon:
            curr_id = row['id']
            idx_in_src_shp = source_shapefiles_geo[src_shp][source_shapefiles_geo[src_shp]['class_name'] == curr_id].index
            if idx_in_src_shp.empty:
                warnings.warn(f"ID {curr_id} not found in shapefile ({shapefile_mapper[src_shp][0]}). Fake polygon will be created but please, check the reference points CSV.")
                create_fake_polygon = True
            elif len(idx_in_src_shp) > 1:
                raise ValueError(f"Multiple IDs ({curr_id}) found in shapefile ({shapefile_mapper[src_shp][0]}). Please check the reference points CSV.")
        
        if create_fake_polygon:
            curr_proj_x = row['proj_x']
            curr_proj_y = row['proj_y']
            curr_proj_poly = create_polygons_from_points(x=curr_proj_x, y=curr_proj_y, buffer=point_buffer, shape=point_buffer_type)[0]
            curr_geo_poly = convert_polygons_crs(polygons=curr_proj_poly, crs_in=proj_epsg_landslides, crs_out=INTERNAL_EPSG_CODE)[0]
        else:
            curr_geo_poly = source_shapefiles_geo[src_shp].loc[idx_in_src_shp[0], 'geometry']
            if not isinstance(curr_geo_poly, (geom.Polygon, geom.MultiPolygon)):
                create_fake_polygon = True
                curr_geo_poly = curr_geo_poly.buffer(1e-8) # Add a small buffer
                curr_proj_poly = convert_polygons_crs(polygons=curr_geo_poly, crs_in=INTERNAL_EPSG_CODE, crs_out=proj_epsg_landslides)[0].buffer(point_buffer)
                curr_geo_poly = convert_polygons_crs(polygons=curr_proj_poly, crs_in=proj_epsg_landslides, crs_out=INTERNAL_EPSG_CODE)[0]
            else:
                curr_proj_poly = convert_polygons_crs(polygons=curr_geo_poly, crs_in=INTERNAL_EPSG_CODE, crs_out=proj_epsg_landslides)[0]
        
        landslides_df.loc[idx, 'geometry'] = curr_geo_poly
        landslides_df.loc[idx, 'area_in_sq_meters'] = curr_proj_poly.area
        landslides_df.loc[idx, 'generated_from_point'] = create_fake_polygon
    
    landslides_gdf = gpd.GeoDataFrame(landslides_df, geometry='geometry', crs=f"EPSG:{INTERNAL_EPSG_CODE}")

    if write_out_shapefile:
        if out_shapefile_filename is None:
            out_shapefile_filename = f"pslip_landslides_epsg_{out_shapefile_epsg_code}.shp"
        if not out_shapefile_filename.endswith('.shp'):
            out_shapefile_filename += '.shp'
        
        out_shapefile_path = os.path.join(env.folders['inputs'][src_type]['path'], out_shapefile_filename)
        landslides_gdf.to_crs(epsg=out_shapefile_epsg_code, inplace=False).to_file(out_shapefile_path, driver='ESRI Shapefile')

    landslides_vars = {
        'geodataframe': landslides_gdf,
        'current_epsg': INTERNAL_EPSG_CODE,
        'proj_epsg': proj_epsg_landslides,
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
    args = parser.parse_args()

    study_area_vars = main(
        base_dir=args.base_dir, 
        gui_mode=args.gui_mode
    )

# %%
