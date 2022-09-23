#author: jun wang @ iee

import bpy
import bmesh
import json
import os
import numpy as np
from bpy import context
import builtins as __builtin__

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


def print_vert_details(selected_verts):
    # number of vertices
    num_verts = len(selected_verts)
    print("Number of verticies: {}".format(num_verts))

    print("Vertex id: {}".format([id.index for id in selected_verts]))
    #print("Vertex coordinates: {}".format([id.co[:] for id in selected_verts]))

def get_vertex_data(object_reference):
    # get the mesh in edit mode
    bm = bmesh.from_edit_mesh(object_reference.data)
    # get all the selected vertices
    selected_verts = [vert for vert in bm.verts if vert.select]
    # print the result
    #print_vert_details(selected_verts)
    return selected_verts
    
def where_file(file_name):
    return os.path.exists(file_name)

def save_verts(keyname, dst_file):
    print("--keyname--: ", keyname)  
    object_reference = bpy.context.active_object
    bm = bmesh.from_edit_mesh(object_reference.data)
    verts = [vert for vert in bm.verts if vert.select]
    print_vert_details(verts)

    records = {} 
    for vid in verts:
        records[vid.index] = np.array(vid.co[:]) 
    if not where_file(dst_file):
        data = {keyname: records}
        np.save(dst_file, data)
    else:
        data = np.load(dst_file)
        data = data.item()
        print(data.keys())

        #todo: simplify this if-else:
        if not keyname in data:
            data[keyname] =  records
            np.save(dst_file, data)
        else:
            print("keyname: ", keyname, " has been used. overwrite...")
            data[keyname] =  records
            np.save(dst_file, data)
    
def where_json(file_name):
    return os.path.exists(file_name)
        
def save_meshname(keyname, dst_file):
    selected_obj = bpy.context.selected_objects[0]
    objname = selected_obj.name
    print("---",keyname, "---name--- ", objname)
    if not where_file(dst_file):
        data = {keyname: objname}
        np.save(dst_file, data)
    else:
        data = np.load(dst_file)
        data = data.item()
        print(data.keys())

        if keyname in data:
            print("keyname: ", keyname, " has been used. overwrite...")
        data[keyname] =  objname
        np.save(dst_file, data)

#label key ID matchs the 68 facial landmarks make up
label_key = [[18, 22], #right eyebow
             [23, 27], # left eyebow
             [28,31],  # nose ridge
             [32,34,36], # nose
             [37,38,39,40,41,42,69], #right eye
             [43,44,45,46,47,48,70], # left eye
             [49,52,55, 58]] # mouth 

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#

#not sure how blender arrange its vertex id, so we have to label one by one
model_name = "aab01_o"
#digital/meshname/eyename
keyname = 31

#configure your own labelling file
#windows OS
dst_file = "C:\\jwang\\mh_label\\"+model_name+"_label.npy"

if isinstance(keyname, int):
    save_verts(keyname, dst_file)
else:
    save_meshname(keyname, dst_file)


