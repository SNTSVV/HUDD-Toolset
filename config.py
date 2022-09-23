import os
# components = ["mouth", "noseridge", "nose", "rightbrow", "righteye", "lefteye", "leftbrow"]
components = ["mouth"]

## Mac-Laptop:
#blenderPath = "/Applications/Blender.app/Contents/MacOS/blender"

## Mac-Desktop:
#blenderPath = "/Applications/Blender.app/Contents/MacOS/blender"

## HPC:
#blenderPath = "/home/users/hfahmy/blender-2.79/blender" #iee_version == 1
# blenderPath_2 = "/home/users/hfahmy/blender-2.81/blender" #iee_version == 2

## Windows-Laptop:
#blenderPath = "blender" #IEE_V2 #Blender2.81+
blenderPath = "C:\Program Files\Blender Foundation\Blender2.79\Blender" #IEE_V1 #Blender2.79

#nVar = 20 #IEE_V1 (Constant Parameters)
nVar = 13 #IEE_V1
#nVar = 23 #IEE_V2
indvdSize = 1
globalCounter = 0
BL = False

#iee_version == 1:
width = 752 // 2  # 376
height = 480 // 2  # 240

#iee_version == 2:
#width=
#height=