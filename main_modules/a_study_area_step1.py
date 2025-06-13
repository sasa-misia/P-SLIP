import os
import pickle
import numpy as np
import shapely.geometry as geom
import shapely.ops as ops
import geopandas as gpd
from main_modules._analysis_init import load_or_create_analysis
import pandas as pd

def study_area_step1(
    file_name_study_area,
    mun_field_name,
    mun_sel,
    specific_window,
    case_name=None,
    base_dir=None,
    earth_radius=6371000
):
    """
    Defines the study area and saves the main variables.
    Loads folder paths from the analysis environment and input_files.csv.
    """
    # Load analysis environment
    env = load_or_create_analysis(case_name=case_name, base_dir=base_dir)
    fold_var = env.var_dir['path']

    # Find fold_raw_mun from input_files.csv where type == 'raw_mun'
    input_files_path = os.path.join(env.inp_dir['path'], 'input_files.csv')
    if not os.path.exists(input_files_path):
        raise FileNotFoundError(f"input_files.csv not found at {input_files_path}")
    input_files_df = pd.read_csv(input_files_path)
    raw_mun_row = input_files_df[input_files_df['type'] == 'raw_mun']
    if raw_mun_row.empty:
        raise ValueError("No entry with type 'raw_mun' found in input_files.csv")
    if len(raw_mun_row) > 1:
        raise ValueError("Multiple entries with type 'raw_mun' found in input_files.csv. Please ensure only one exists.")
    # Use the only match
    raw_mun_path = raw_mun_row.iloc[0]['path']
    # If the path is relative, make it absolute with respect to the inputs folder
    if not os.path.isabs(raw_mun_path):
        fold_raw_mun = os.path.abspath(os.path.join(env.inp_dir['path'], raw_mun_path))
    else:
        fold_raw_mun = raw_mun_path

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
        print("Creation of specific window...")
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
    Loads shapefile with geopandas, filters and returns a list of shapely Polygons and MunSel.
    """
    gdf = gpd.read_file(shapefile_path)
    if sel_filter is not None and len(sel_filter) > 0:
        # sel_filter can be numpy array or list
        sel_values = set([str(s[0]) if isinstance(s, (np.ndarray, list)) else str(s) for s in sel_filter])
        gdf = gdf[gdf[field_name].astype(str).isin(sel_values)]
    polygons = [geom.shape(geom_) if not isinstance(geom_, geom.Polygon) else geom_ for geom_ in gdf.geometry]
    mun_sel = gdf[field_name].tolist()
    return polygons, mun_sel
