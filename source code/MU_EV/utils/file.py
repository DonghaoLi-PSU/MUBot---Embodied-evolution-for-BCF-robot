# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:49:17 2023

@author: dongh
"""

import json, math, os, pickle, re, yaml,pathlib, shutil
import pandas as pd
from lxml import etree

from MU_EV.config import cfg
from MU_EV.utils import xml as xu

def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)

def check_folder(path,worker_ID,morph_ID):
    if not os.path.exists(path+"/Database/"+str(worker_ID)):
        os.mkdir(path+"/Database/"+str(worker_ID))
    if not os.path.exists(path+"/Population_data/"+str(morph_ID)):
        os.mkdir(path+"/Population_data/"+str(morph_ID))

def df_to_csv(df,path,file_type,worker_ID,morph_ID):
    check_folder(path,worker_ID,morph_ID)
    df.to_csv(path+"/Database/"+str(worker_ID)+"/"+file_type+"_param.csv",header=False)
    df.to_csv(path+"/Population_data/"+str(morph_ID)+"/"+file_type+"_param.csv",header=False)

def urdf_write(root,path,worker_ID,morph_ID):
    check_folder(path,worker_ID,morph_ID)
    tree = etree.ElementTree(root)
    tree.write(path+"/Population_data/"+str(morph_ID)+"/"+"mubot.urdf",xml_declaration=True,encoding='utf-8', pretty_print=True)
    tree.write(path+"/Database/"+str(worker_ID)+"/"+"mubot.urdf",xml_declaration=True,encoding='utf-8', pretty_print=True)

def move_files(src_folder,dst_folder):
    src_path = pathlib.Path(src_folder)
    dst_path = pathlib.Path(dst_folder)
    if not os.path.exists(dst_folder):
        print(f"Destination folder {dst_folder} does not exist. Creating it.")
    for item in src_path.iterdir():
            destination_path = dst_path / item.name
            try:
                shutil.move(str(item), str(destination_path))
            except shutil.Error as e:
                print(f"Error moving {item.name}: {e}")
