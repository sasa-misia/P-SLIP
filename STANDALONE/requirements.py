# -*- coding: utf-8 -*-
"""
Slip & Evolution - Landslides predictor

@author: salva
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import platform
import pickle
import os
import tkinter as tk; from tkinter import filedialog, ttk
import shapely.geometry as shg; from shapely.geometry import Polygon
import datetime as date
import shutil
import pyproj as prj
from openpyxl import load_workbook

#%% Reloading of paths
def folders_creation():
    if platform.system().lower() == 'darwin': sl = '/'
    elif platform.system().lower() == 'windows': sl = '\\'
    else: print('Platform not supported!')
    Folders = dict()
    Folders['main'] = os.getcwd()
    Folders['raw'] = Folders['main']+sl+'Raw Data'
    Folders['raw_rain'] = Folders['raw']+sl+'Rainfalls'
    Folders['raw_det_ss'] = Folders['raw']+sl+'Detected Soil Slipes'
    Folders['raw_mun'] = Folders['raw']+sl+'Municipalities'
    Folders['raw_lit'] = Folders['raw']+sl+'Lithology'
    Folders['raw_sat'] = Folders['raw']+sl+'Satellite Images'
    Folders['raw_road'] = Folders['raw']+sl+'Roads'
    Folders['raw_land_uses'] = Folders['raw']+sl+'Land Uses'
    Folders['raw_dtm'] = Folders['raw']+sl+'DTM'
    Folders['raw_veg'] = Folders['raw']+sl+'Vegetation'
    Folders['var'] = Folders['main']+sl+'Variables'
    Folders['res'] = Folders['main']+sl+'Results'
    Folders['res_fs'] = Folders['res']+sl+'Factors of Safety'
    Folders['res_flow'] = Folders['res']+sl+'Flow Paths'
    Folders['user'] = Folders['main']+sl+'User Control'
    Folders['fig'] = Folders['main']+sl+'Figures'
    # Creating folders
    for i in Folders.keys():
        if os.path.isdir(Folders[i]) != 1:
            os.mkdir(Folders[i])
    # Saving
    file = open('os_folders.pickle', 'wb')
    pickle.dump([Folders, sl], file)
    file.close()
    
if not os.path.exists('os_folders.pickle'): folders_creation()
folders_file = open('os_folders.pickle', 'rb')
FoldersTemp, _ = pickle.load(folders_file) # If there are multiple output to unpack and not only 2 you can use FolderTemp, *_
folders_file.close()
if os.getcwd() != FoldersTemp['main']: folders_creation()
folders_file = open('os_folders.pickle', 'rb')
Folders, sl = pickle.load(folders_file)
del folders_file, FoldersTemp

#%% Program Definitions
#### ~uiopenfile~
def uiopenfile(ft=[('General', '*.*')], idir=os.getcwd()):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath = filedialog.askopenfilename(title='Select file', filetypes=ft, initialdir=idir)
    dirfilepath = os.path.dirname(filepath)
    totfilename = os.path.basename(filepath)
    filename = totfilename.split('.')[0]
    if filepath != idir and not(os.path.exists(f"{idir}{sl}{totfilename}")):
        for currfile in os.listdir(dirfilepath): 
            if currfile.split('.')[0] == filename: 
                shutil.copy(f"{dirfilepath}{sl}{currfile}", idir)
        newfilepath = f"{idir}{sl}{totfilename}"
    else:
        newfilepath = filepath
    root.destroy()
    return newfilepath


#### ~uioptionmenu~
def uioptionmenu(opts=['no values'], multi=False): # You have to give a List with opts!
    root = tk.Tk()
    root.title('Option menu')
    # root.iconbitmap('c:/') # You have to tell the path of the icon you want
    root.geometry('400x650')
    root.config(bg='pink')
    root.attributes("-topmost", True)
    value = tk.StringVar(root)
    value.set("Select an Option")
    # dropdown = tk.OptionMenu(root, value, *opts)
    # dropdown.pack(expand=True)
    dropdown = ttk.Combobox(root, values=opts, textvariable=value, state='readonly', width = 15)
    dropdown.pack(expand=True)
    submit = tk.Button(root, text='Submit', command=root.destroy)
    submit.pack(side=tk.BOTTOM, pady=30)
    # Multi choice menu
    if multi:
        def autosearch(event):
            srchres = [x.lower().startswith(event.char.lower()) for x in opts]
            try: 
                frstresind = [i for i, x in enumerate(srchres) if x][0]
                value.set(opts[frstresind])
            except IndexError:
                frstresind = 0
            
        def selall():
            global selallclick
            selallclick = True
            root.destroy()
            
        def cancel(self):
            if self in globals():
                globals()[self].pop()
                
        def refresh(self):
            if self in globals():
                globals()[self][-1].destroy()
                globals()[self].pop()
            
        def selectfrom(event=None): # None default is necessary because Button don't pass an input in contrast to bind
            global sellabel, labeltext
            if (0<=event.x<=selfromlist.winfo_width() and 
                0<=event.y<=selfromlist.winfo_height()):
                if 'sellabel' not in globals(): 
                    sellabel = list()
                    labeltext = list()
                labeltext.append(tk.Label(root, text=value.get()))
                labeltext[-1].pack()
                sellabel.append(value.get())
        # Select all button
        selectall = tk.Button(root, text='Select all', command=selall)
        selectall.pack(side=tk.BOTTOM, pady=30)
        # Remove button
        cancellist = tk.Button(root, text='Remove last element', 
                               command=lambda: [cancel('sellabel'), refresh('labeltext')])
        cancellist.pack(side=tk.BOTTOM, pady=30)
        # Selection button
        selfromlist = tk.Button(root, text='Add to selection')
        selfromlist.bind('<ButtonRelease-1>', selectfrom)
        selfromlist.unbind('<Leave>')
        selfromlist.pack(side=tk.BOTTOM, pady=30)
        # Assign event key to dropdown for search
        dropdown.bind('<Key>', autosearch) # If I change with bind_all even if not selected it will searchs
        dropdown.bind('<<ComboboxSelected>>', selectfrom)
    # Returned in the main loop
    # root.bind_all("<Enter>", lambda event:event.widget.focus_set())
    root.mainloop()
    # Choose what value to return
    if multi and ('selallclick' in globals()):
        selection = opts
    if multi and ('sellabel' in globals()):
        selection = list(set(sellabel))
    elif multi == False:
        selection = value.get()
    # Return value
    return selection


#### ~varload~
def varload(filename='untitled', path=Folders['var']):
    file = open(f"{path}{sl}{filename}.pickle", 'rb')
    outputvar = pickle.load(file)
    file.close()
    return outputvar
 
    
#### ~varsave~
def varsave(variables, filename='untitled', path=Folders['var']):
    file = open(f"{path}{sl}{filename}.pickle", 'wb')
    pickle.dump(variables, file)
    file.close()
    
    
#### ~checkplot~
def checkplot(shape, poly, save=False, figtitle='untitled', savepath=Folders['fig']):
    Fig, Ax = plt.subplots(figsize=(12,10))
    Fig.canvas.manager.set_window_title('Check Plot')
    if poly.type == 'Polygon':
        Ax.plot(*poly.exterior.xy, c = 'black', linewidth=1.6)
    elif poly.type == 'MultiPolygon':
        for singlepoly in poly:
            Ax.plot(*singlepoly.exterior.xy, c = 'black', linewidth=1.6)
    shape.plot(ax=Ax, color='pink', alpha=0.5)
    Ax.set(xlabel='Long (°)', ylabel='Lat (°)', title=figtitle)
    Ax.set_aspect('equal')
    Fig.show()
    if save:
        Fig.savefig(f"{savepath}{sl}{figtitle}.png", dpi=300)


#### ~prj_lat_to_CRS~
def bbox_lat2CRS(pathCRS, bboxWGS):
    BbLong = bboxWGS.values[0][[0,2]]; BbLat = bboxWGS.values[0][[1,3]]
    DestCRS = gpd.read_file(pathCRS, rows=1).crs
    Bb = prj.Transformer.from_crs("epsg:4326", DestCRS).transform(BbLat, BbLong)
    Bounding = (Bb[0][0], Bb[1][0], Bb[0][1], Bb[1][1])
    return Bounding


#### ~extract_coord_num~
def coord_num(geom):
    if geom.type == 'Polygon':
        exterior_coords = len(geom.exterior.coords)
        interior_coords = 0
        for interior in geom.interiors:
            interior_coords += len(interior.coords)
        elem_sum_coords = exterior_coords+interior_coords
    elif geom.type == 'MultiPolygon':
        elem_sum_coords = 0
        for part in geom:
            epc = coord_num(part)  # Recursive call
            elem_sum_coords += epc
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    return elem_sum_coords


#### ~update_excel~
# def update_excel(filepath, df):
#     FilePath = "your excel path"
#     ExcelWorkbook = load_workbook(FilePath)
#     writer = pd.ExcelWriter(FilePath, engine = 'openpyxl')
#     writer.book = ExcelWorkbook
#     df.to_excel(writer, sheet_name = 'your sheet name')
#     writer.save()
#     writer.close()