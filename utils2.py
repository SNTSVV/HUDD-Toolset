"""
Author: Jun WANG @ IEE
"""

import os
from os import listdir
from os.path import isfile, join, isdir
import builtins as __builtin__
from bpy import context

def ensure_folder(folder_name):
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)

def get_files(folder, extx=".csv"):
    all_files = []
    print("getting .csv in ", folder)
    for f in listdir(folder):
        xf = join(folder, f)
        if isfile(xf) and xf.endswith(extx):
            all_files.append(xf)
    return all_files

def get_model_label_files(folder, model_type=".blend", label_type=".npy"):
    """
    Get a list of (model file, label file) pairs.

    Parameters
    ----------
    folder: pl.Path:
        The path to the folder containing the model and label files.
    model_type: str
        Extension of the model files.
    label_type: str
        Extension of the label files.
    
    Return
    ------
    A list of model and label file pairs.
    """
    print("looking for model in", folder)
    all_fills = []
    for f in listdir(folder):
        xfold = join(folder,f)
        if isdir(xfold):
            model_file, label_file = None, None
            model_num, label_num = 0, 0
            for sub in listdir(xfold):
                subf = join(xfold,sub)
                if isfile(subf):
                    if subf.endswith(model_type):
                        model_file = subf
                        model_num +=1
                    elif subf.endswith(label_type):
                        label_file = subf
                        label_num +=1
            #if model_num==1 and label_num==1:
            if model_file and label_file:
                all_fills.append([model_file,label_file])
    print("model label files", all_fills)
    return all_fills


def console_print(*args, **kwargs):
    for a in context.screen.areas:
        if a.type == 'CONSOLE':
            c = {}
            c['area'] = a
            c['space_data'] = a.spaces.active
            c['region'] = a.regions[-1]
            c['window'] = context.window
            c['screen'] = context.screen
            s = " ".join([str(arg) for arg in args])
            for line in s.split("\n"):
                bpy.ops.console.scrollback_append(c, text=line)

def print(*args, **kwargs):
    """Console print() function."""
    console_print(*args, **kwargs) # to py consoles
    __builtin__.print(*args, **kwargs) # to system console