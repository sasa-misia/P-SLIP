# -*- coding: utf-8 -*-
"""
Slip & Evolution - Landslides predictor

@author: salva
"""

from requirements import *

_, StudyAreaPolygon, StudyAreaPolygonClean, StudyAreaShape, StudyBbox = varload('StudyAreaVariables')

FilepathLndUse = uiopenfile(idir=Folders['raw_land_uses'], ft=[('sapefile', '*.shp')])

Bounding = bbox_lat2CRS(FilepathLndUse, StudyBbox)

LndUseShape = gpd.read_file(FilepathLndUse, Bounding)
# PointNumInFields = [len(x.exterior.coords) for x in LndUseShape['geometry']] # To count element in each row and remove too big element later
PointNumInFields = [coord_num(x) for x in LndUseShape['geometry']]

LndUseShapeFields = LndUseShape.columns.unique().append(pd.Index(['None of these'])).to_list()
LndUseShapeFields.sort()
LndUseFieldName = uioptionmenu(LndUseShapeFields)

LndUseOpt = pd.unique(LndUseShape[LndUseFieldName]).tolist()
LndUseOpt.sort()
LndUseSel = uioptionmenu(LndUseOpt, multi=True)
LndUseShapeSelection = LndUseShape.loc[LndUseShape[LndUseFieldName].isin(LndUseSel)
                                       ].dissolve(by=LndUseFieldName).to_crs(epsg=4326)

LndUseShapeSelection.drop(
    LndUseShapeSelection.index[~LndUseShapeSelection['geometry'].intersects(StudyAreaPolygon)],
    inplace=True)

# LndUseShapeSelection = LndUseShapeSelection['geometry'].intersection(StudyAreaPolygon)
# LndUseShapeSelection = LndUseShapeSelection.loc[~LndUseShapeSelection.is_empty]

FilepathLndUseAss = f"{Folders['user']}{sl}LandUsesAssociation.xlsx"
ColHeader1 = [LndUseFieldName,'Abbreviation (for Map)','RGB (for Map)']
ExcelToWrite = pd.DataFrame({LndUseFieldName : LndUseOpt,
                             'Abbreviation (for Map)' : str(), # Maybe is better to make empty values
                             'RGB (for Map)' : '0 0 0'}) # Maybe is better to make empty values
ExcelToWrite.to_excel(FilepathLndUseAss, sheet_name='Association', index=False)

varsave([FilepathLndUse, LndUseFieldName, LndUseSel], filename='User_A_LandUses')
varsave(LndUseShapeSelection, filename='LandUsesVariables')