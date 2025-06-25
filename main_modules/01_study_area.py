import os
import pickle
import numpy as np
import shapely.geometry as geom
import shapely.ops as ops
import geopandas as gpd
from config.analysis_init import get_analysis_environment
import pandas as pd
from psliptools.utilities.pathutils import get_raw_path  # Import from pslip toolbox

def study_area_step1(
    file_name_study_area,
    mun_field_name,
    mun_sel,
    specific_window,
    analysis_dir,
    earth_radius=6371000
):
    """
    Defines the study area and saves the main variables.
    Loads folder paths from the analysis environment and input_files.csv.
    """
    # Load analysis environment
    env = get_analysis_environment(base_dir=analysis_dir)
    fold_var = env.var_dir['path']

    # Get the absolute path to the raw municipality shapefile
    fold_raw_mun = get_raw_path(env.inp_dir['path'], 'raw_mun')

    sl = env.os_separator if hasattr(env, 'os_separator') else os.sep
    study_area_polygon_excluded = geom.Polygon()

    if file_name_study_area == 'None of these' and not specific_window:
        raise ValueError("If there is no shapefile (None of these), then you must select the Win checkbox!")

    # Options for specific window
    choice_window = 'SingleWindow'
    info_detected_soil_slips = None
    ind_def_info_det = None
    info_soil_slip_path = os.path.join(fold_var, 'InfoDetectedSoilSlips.pkl')
    if specific_window:
        if os.path.exists(info_soil_slip_path):
            with open(info_soil_slip_path, 'rb') as f:
                data = pickle.load(f)
                info_detected_soil_slips = data['InfoDetectedSoilSlips']
                ind_def_info_det = data['IndDefInfoDet']
            opt = input(
                "Would you like to create a single window or multiple windows based on detected soil slip? (SingleWindow/MultiWindows): "
            ).strip()
            if opt in ['SingleWindow', 'MultiWindows']:
                choice_window = opt

    # MunPolygon and MunSel creation
    mun_polygon = None
    if file_name_study_area != 'None':
        st_ar_shape_path = os.path.join(fold_raw_mun, str(file_name_study_area))
        mun_polygon, mun_sel = polyshapes_from_shapefile(
            st_ar_shape_path, mun_field_name, sel_filter=mun_sel, points_lim=500000
        )

    pol_window = None
    if specific_window:
        print("Creating specific window...")
        if choice_window == 'SingleWindow':
            lon_min = float(input("Lon min [°]: "))
            lon_max = float(input("Lon max [°]: "))
            lat_min = float(input("Lat min [°]: "))
            lat_max = float(input("Lat max [°]: "))
            pol_window = geom.Polygon([
                (lon_min, lat_min),
                (lon_max, lat_min),
                (lon_max, lat_max),
                (lon_min, lat_max)
            ])
        elif choice_window == 'MultiWindows':
            if info_detected_soil_slips is None or ind_def_info_det is None:
                raise RuntimeError("InfoDetectedSoilSlips data not loaded.")
            x_lon_det = [row[4] for row in info_detected_soil_slips[ind_def_info_det]]
            y_lat_det = [row[5] for row in info_detected_soil_slips[ind_def_info_det]]
            wnd_side = float(input("Side of each window [m] (default 1200): ") or "1200")
            d_lat_half = np.degrees(wnd_side / 2 / earth_radius)
            pol_window = []
            for x, y in zip(x_lon_det, y_lat_det):
                d_lon_half = d_lat_half / np.cos(np.radians(y))
                poly = geom.Polygon([
                    (x - d_lon_half, y - d_lat_half),
                    (x + d_lon_half, y - d_lat_half),
                    (x + d_lon_half, y + d_lat_half),
                    (x - d_lon_half, y + d_lat_half)
                ])
                pol_window.append(poly)
        # Intersection with MunPolygon
        if mun_polygon is not None:
            if isinstance(pol_window, list):
                union_window = ops.unary_union(pol_window)
            else:
                union_window = pol_window
            mun_polygon = [mp.intersection(union_window) for mp in mun_polygon]
            # Remove empty polygons
            mun_polygon = [mp for mp in mun_polygon if not mp.is_empty]
        else:
            if isinstance(pol_window, list):
                mun_sel = [f"Poly {i+1}" for i in range(len(pol_window))]
                mun_polygon = pol_window
            else:
                mun_sel = ["Poly 1"]
                mun_polygon = [pol_window]

    # MunSel as column array
    mun_sel = np.array(mun_sel).reshape(-1, 1)

    # Union of polygons
    print("Union of polygons...")
    study_area_polygon = ops.unary_union(mun_polygon)
    study_area_polygon_clean = study_area_polygon

    # Study area limits
    if hasattr(study_area_polygon, "exterior"):
        xs, ys = study_area_polygon.exterior.xy
        max_extremes = (max(xs), max(ys))
        min_extremes = (min(xs), min(ys))
    else:
        max_extremes = (None, None)
        min_extremes = (None, None)

    # Saving
    print("Finishing...")
    vars_study_area = {
        'mun_polygon': mun_polygon,
        'study_area_polygon': study_area_polygon,
        'study_area_polygon_clean': study_area_polygon_clean,
        'study_area_polygon_excluded': study_area_polygon_excluded,
        'max_extremes': max_extremes,
        'min_extremes': min_extremes
    }
    if specific_window:
        vars_study_area['pol_window'] = pol_window
    vars_user_study = {
        'file_name_study_area': file_name_study_area,
        'mun_field_name': mun_field_name,
        'mun_sel': mun_sel,
        'specific_window': specific_window
    }
    with open(os.path.join(fold_var, 'StudyAreaVariables.pkl'), 'wb') as f:
        pickle.dump(vars_study_area, f)
    with open(os.path.join(fold_var, 'UserStudyArea_Answers.pkl'), 'wb') as f:
        pickle.dump(vars_user_study, f)

def polyshapes_from_shapefile(shapefile_path, field_name, sel_filter=None, points_lim=500000):
    """
    Loads a shapefile with geopandas, filters and returns a list of shapely Polygons and MunSel.
    """
    gdf = gpd.read_file(shapefile_path)
    if sel_filter is not None and len(sel_filter) > 0:
        # sel_filter can be numpy array or list
        sel_values = set([str(s[0]) if isinstance(s, (np.ndarray, list)) else str(s) for s in sel_filter])
        gdf = gdf[gdf[field_name].astype(str).isin(sel_values)]
    polygons = [geom.shape(geom_) if not isinstance(geom_, geom.Polygon) else geom_ for geom_ in gdf.geometry]
    mun_sel = gdf[field_name].tolist()
    return polygons, mun_sel

def main(
    file_name_study_area=None,
    mun_field_name=None,
    mun_sel=None,
    specific_window=False,
    case_name=None,
    base_dir=None,
    earth_radius=6371000,
    gui_mode=False
):
    """
    Main entrypoint for study area step. Can be called from CLI, GUI, or as a module.
    """
    # If not in GUI mode, prompt for missing arguments
    if not gui_mode:
        if file_name_study_area is None:
            file_name_study_area = input("Shapefile name for study area (or 'None'/'None of these'): ")
        if mun_field_name is None:
            mun_field_name = input("Field name for municipality selection: ")
        if mun_sel is None:
            mun_sel = input("Municipality selection (comma separated): ").split(",")
        if specific_window is None:
            specific_window = input("Specific window? [y/N]: ").strip().lower() == "y"
        if case_name is None:
            case_name = input("Case name (optional): ") or None
        if base_dir is None:
            base_dir = input("Base directory (optional): ") or None

    study_area_step1(
        file_name_study_area=file_name_study_area,
        mun_field_name=mun_field_name,
        mun_sel=mun_sel,
        specific_window=specific_window,
        case_name=case_name,
        base_dir=base_dir,
        earth_radius=earth_radius
    )
    
    if not gui_mode:
        print("Study area step completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Define the study area and save main variables.")
    parser.add_argument("--file_name_study_area", type=str, help="Shapefile name for study area (or 'None'/'None of these')")
    parser.add_argument("--mun_field_name", type=str, help="Field name for municipality selection")
    parser.add_argument("--mun_sel", type=str, nargs='*', help="Municipality selection (space separated)")
    parser.add_argument("--specific_window", action="store_true", help="Use specific window")
    parser.add_argument("--case_name", type=str, default=None, help="Case name")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory")
    parser.add_argument("--earth_radius", type=float, default=6371000, help="Earth radius in meters")
    args = parser.parse_args()

    main(
        file_name_study_area=args.file_name_study_area,
        mun_field_name=args.mun_field_name,
        mun_sel=args.mun_sel,
        specific_window=args.specific_window,
        case_name=args.case_name,
        base_dir=args.base_dir,
        earth_radius=args.earth_radius,
        gui_mode=False
    )