import os
import json
import shapely.geometry as geom
import shapely.ops as ops
import geopandas as gpd

def import_land_use_shapefile(env, shapefile_path, landuse_field):
    """Import land use shapefile and update env with land use info."""
    gdf = gpd.read_file(shapefile_path)
    unique_landuse = gdf[landuse_field].unique().tolist()
    # Save only the info needed for env (no objects)
    env['land_use'] = {
        'shapefile_path': shapefile_path,
        'landuse_field': landuse_field,
        'unique_landuse': unique_landuse,
        'imported': True
    }
    # Initialize removed_areas if not present
    if 'removed_areas' not in env:
        env['removed_areas'] = []
    return gdf, unique_landuse, env

def import_vegetation_shapefile(env, shapefile_path, veg_field):
    """Placeholder for vegetation shapefile import."""
    pass

def import_other_shapefile(env, shapefile_path, field_name):
    """Placeholder for other shapefile import."""
    pass

def save_env(env, env_path):
    with open(env_path, 'w', encoding='utf-8') as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

def main():
    # Example usage
    env_path = "./Variables/env.json"
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            env = json.load(f)
    else:
        env = {}

    shapefile_path = input("Land use shapefile path: ")
    landuse_field = input("Land use field name: ")

    gdf, unique_landuse, env = import_land_use_shapefile(env, shapefile_path, landuse_field)
    save_env(env, env_path)
    print("Land use imported. Unique classes:", unique_landuse)

if __name__ == "__main__":
    main()
