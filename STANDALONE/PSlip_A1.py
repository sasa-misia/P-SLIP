# -*- coding: utf-8 -*-
"""
Slip & Evolution - Landslides predictor

@author: salva
"""

from requirements import *

FilepathStudyArea = uiopenfile(idir=Folders['raw_mun'], ft=[('sapefile', '*.shp')])

MunShape = gpd.read_file(FilepathStudyArea)
MunShapeFields = MunShape.columns.unique().append(pd.Index(['None of these'])).to_list()
MunShapeFields.sort()
MunFieldName = uioptionmenu(MunShapeFields)

if MunFieldName != "None of these":
    MunOpt = MunShape[MunFieldName].to_list()
    MunOpt.sort()      
    MunSel = uioptionmenu(MunOpt, multi=True)
    MunShapeSelection = MunShape.loc[MunShape[MunFieldName].isin(MunSel)]
elif MunFieldName == "None of these":
    MunSel = [];
    MunFieldName = [];
    MunShapeSelection = MunShape
    
MunShapeSelection = MunShapeSelection.to_crs(epsg=4326)

StudyAreaPolygon = MunShapeSelection.unary_union
StudyAreaPolygonClean = StudyAreaPolygon # Maybe is better if import copy and made a copy.deepcopy(OriginalObject)
StudyShapeInfo = {'Study Area Composition': [MunSel], 
                  'Creation Datetime': date.datetime.now().strftime("%d-%B-%Y, %H:%M:%S")}
StudyAreaShape = gpd.GeoDataFrame(StudyShapeInfo, crs=4326, geometry=[StudyAreaPolygon])
StudyBbox = StudyAreaShape.bounds

checkplot(StudyAreaShape, StudyAreaPolygon, save='yes', figtitle='Study Area in Emilia-Romagna')

#### ~Saving~
varsave([FilepathStudyArea, MunFieldName, MunSel], filename='User_A')
varsave([MunShapeSelection, StudyAreaPolygon, StudyAreaPolygonClean, StudyAreaShape, StudyBbox], 
        filename='StudyAreaVariables')