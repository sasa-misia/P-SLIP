import os
import json
import geopandas as gpd

def remove_landuse_areas(env, to_remove):
    """Update env with removed land use classes."""
    if 'removed_areas' not in env:
        env['removed_areas'] = []
    env['removed_areas'].extend(to_remove)
    # Remove duplicates
    env['removed_areas'] = list(sorted(set(env['removed_areas'])))
    return env

def save_env(env, env_path):
    with open(env_path, 'w', encoding='utf-8') as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

def main():
    env_path = "./Variables/env.json"
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            env = json.load(f)
    else:
        print("No env found. Please import land use first.")
        return

    if 'land_use' not in env or not env['land_use'].get('imported', False):
        print("Land use not imported yet.")
        return

    unique_landuse = env['land_use']['unique_landuse']
    print("Available land use classes:", unique_landuse)
    to_remove = input("Land use classes to remove (comma separated): ").split(",")
    to_remove = [x.strip() for x in to_remove if x.strip() in unique_landuse]
    if not to_remove:
        print("No valid classes selected.")
        return

    env = remove_landuse_areas(env, to_remove)
    save_env(env, env_path)
    print("Removed areas updated in env:", env['removed_areas'])

if __name__ == "__main__":
    main()
