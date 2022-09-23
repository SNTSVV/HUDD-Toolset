#author: jun wang @ iee
from pathlib import Path


#model_type opitons: ".blend", ".obj", ".mhx2"
model_type = ".blend" # For 4d-face put .obj here
constain_face=45
muti_processor = False
GPUs = [0] #ids for GPU to use. GPUs=[] means disable GPU

# org_type options: 1: 3dmodel files and labels are seperated located; 2: 3dmodel file and its label are in one folder
org_type = 2 # For 4dface put 1 here

if model_type==".blend":
    model_folder = "C:/Users/hazem.fahmy/Documents/HPD/TR/Pool/"
    label_folder = None
elif model_type==".obj":
    model_folder = "./data/new4dfacemodels"
    label_folder = "./data/new4dfacelabels"
elif model_type==".mhx2":
    model_folder = "./data/mhx2"
    label_folder = "./data/newlabel3d"
else:
    model_folder = None
    label_folder = None

pose_folder = "./data/pose"
#width = 1920 #752//2 #376
width = 376 #752//2 #376
#height = 1080 #480//2 #240
height = 240 #480//2 #240
#activate_occlusion = True
log_file = "./data/logs.log"

subsampling = 1

# use this if you want no backgrounds
background_dir = None
# use this if you want backgrounds
#background_dir = "C:/Users/hazem.fahmy/Documents/fabrizio/snt_simulator/data/hdr_background"

"""
sample type:
1: basic #4dface only support type 1
2: put a hand on face, currently, only support model_type==".mhx2"
3: basic/hand on face + different color/energy
4: not supported yet
"""
sample_type = 1 #now only tested on 4dface, makehuman model may have some bugs
#if sample_type==4: activate_occlusion = True

sample_type_string = {1:"basic_test", 2:"hand_on_face", 3:"rnd_env", 4:"ocllusion"}
#dst_datafolder = "./iee_4dface/"+sample_type_string[sample_type]+"/"
#dst_datafolder = "C:/Users/hazem.fahmy/Documents/HPD/TR/Pool/"+model_type.split(".")[-1] +"/"+sample_type_string[sample_type]+"/"


