import os
import sys
from os.path import basename, exists
import math
import bpy
import bpy_extras
import numpy as np

import mathutils
import bmesh
from mathutils.bvhtree import BVHTree
#from bpy.app.handlers import persistent
from mathutils import Euler
import builtins as __builtin__
#bpy.ops.wm.addon_enable(module='import_runtime_mhx2') #IEE_V1
#bpy.ops.wm.save_userpref()
CUDA = False
if CUDA:
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # get the settings (user preferences)
    prefs = bpy.context.preferences.addons['cycles'].preferences #2.81
    #prefs = bpy.context.user_preferences.addons['cycles'].preferences #2.79

    # specify to use CUDA
    prefs.compute_device_type = 'CUDA'

    # check the actual devices installed (the above is only the ones in the preferences)
    cuda_devices, _ = prefs.get_devices()
    print("CUDA", cuda_devices)
    # do not consider the last one (i.e. the CPU)
    # only use half the GPUs, if more than 1 GPU.
    cuda_num = 1
    if len(cuda_devices) > 1:
        cuda_num = len(cuda_devices) // 2
    for i in range(cuda_num):
        cuda_devices[i].use = True

    # save the settings to the profile
    # bpy.ops.wm.save_userpref()
    bpy.ops.wm.save_userpref()

def console_print(*args, **kwargs):
    for a in bpy.context.screen.areas:
        if a.type == 'CONSOLE':
            c = {}
            c['area'] = a
            c['space_data'] = a.spaces.active
            c['region'] = a.regions[-1]
            c['window'] = bpy.context.window
            c['screen'] = bpy.context.screen
            s = " ".join([str(arg) for arg in args])
            for line in s.split("\n"):
                bpy.ops.console.scrollback_append(c, text=line)

def print(*args, **kwargs):
    """Console print() function."""
    console_print(*args, **kwargs) # to py consoles
    __builtin__.print(*args, **kwargs) # to system console



def print_info(msg):
    text_file = open("info.txt", "w")
    text_file.write("INFO: %s \n" % msg)
    text_file.close()

class IEEArmPose(object):
    def __init__(self, bones):
        bpy.ops.object.mode_set(mode='POSE')
        self.bones = bones

    def pose_left_arm_1(self):
        ua101 = self.bones["upperarm01.L"]
        ua102 = self.bones["upperarm02.L"]
        lal01 = self.bones["lowerarm01.L"]


        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"

        axis = 'X'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=95
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

    def pose_left_arm_2(self):
        ua101 = self.bones["upperarm01.L"]
        ua102 = self.bones["upperarm02.L"]
        lal01 = self.bones["lowerarm01.L"]
        lal02 = self.bones["lowerarm02.L"]
        shoulder = self.bones["shoulder01.L"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"

        #--shoulder--
        axis = 'X'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=40
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=40
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=85
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        return

    def pose_left_arm_3(self):
        ua101 = self.bones["upperarm01.L"]
        ua102 = self.bones["upperarm02.L"]
        lal01 = self.bones["lowerarm01.L"]
        lal02 = self.bones["lowerarm02.L"]
        shoulder = self.bones["shoulder01.L"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"
        #--shoulder--
        axis = 'X'
        angle=10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=55+45
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=100
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=30
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=30
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=-50
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        return

    def pose_right_arm_1(self):
        ua101 = self.bones["upperarm01.R"]
        ua102 = self.bones["upperarm02.R"]
        lal01 = self.bones["lowerarm01.R"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"

        axis = 'X'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=95
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

    def pose_right_arm_2(self):
        ua101 = self.bones["upperarm01.R"]
        ua102 = self.bones["upperarm02.R"]
        lal01 = self.bones["lowerarm01.R"]
        lal02 = self.bones["lowerarm02.R"]
        shoulder = self.bones["shoulder01.R"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"

        #--shoulder--
        axis = 'X'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=40
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=40
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=85
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

    def pose_right_arm_3(self):
        ua101 = self.bones["upperarm01.R"]
        ua102 = self.bones["upperarm02.R"]
        lal01 = self.bones["lowerarm01.R"]
        lal02 = self.bones["lowerarm02.R"]
        shoulder = self.bones["shoulder01.R"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"
        #--shoulder--
        axis = 'X'
        angle=10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=55+45
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=100
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=30
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=30
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=-50
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

#creat a person object
#input: file path of makehuman mhx2 model
class IEEPerson(object):
    def __init__(self,filepath):
        self.filepath=filepath
        self.object_name = basename(filepath).split(".")[0]
        self.person_name = self.object_name[:3]
        self.imported_object = self._load_model()

        self.objectx = bpy.data.objects[self.object_name]
        print(self.objectx)
        bpy.ops.object.mode_set(mode='SCULPT')
        print(self.objectx.data.bones)
        self.bones = self.objectx.pose.bones
        self.head_pose = None
        return

    def get_nose_focus(self, shift=(0,0,0)):
        bx = self.bones["special03"].tail
        return np.array(bx)+shift

    def get_breast_focus(self, shift=(0,0,0)):
        bx = (self.bones["breast.R"].tail + self.bones["breast.L"].tail)/2.0
        return np.array(bx)+shift

    def get_forehead_focus(self, shift=(0,0,0)):
        bx = (self.bones["oculi01.R"].tail + self.bones["oculi01.L"].tail)/2.0
        return np.array(bx)+shift


    def _load_model(self):
        # switch of log information
        #---------------------------------------------------------
        logfile = 'blender_load.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        #imported_object = bpy.ops.import_scene.makehuman_mhx2(filepath=self.filepath) #Blender 2.79
        imported_object = bpy.ops.import_scene.obj(filepath=self.filepath)  #Blender 2.81

        os.close(1)
        os.dup(old)
        os.close(old)

        return imported_object

    def adjust_arm(self, idx=None, lamp_type="HEMI"):
        #lamp_type in ["HEMI", "SUN", "POINT", "SPOT", "AREA"]
        armpose = IEEArmPose(self.bones)
        if lamp_type:
            lamp = bpy.data.objects["Lamp"].data
            lamp.type = lamp_type

        if not idx:
            idx = np.random.randint(1,7) #if the "_both_" functions are implemented to (1,11)

        if idx == 1:
            armpose.pose_left_arm_1()
        elif idx == 2:
            armpose.pose_left_arm_2()
        elif idx == 3:
            armpose.pose_left_arm_3()
        elif idx == 4:
            armpose.pose_right_arm_1()
        elif idx == 5:
            armpose.pose_right_arm_2()
        elif idx == 6:
            armpose.pose_right_arm_3()
        elif idx == 7:
            armpose.pose_both_arm_1()
        elif idx == 8:
            armpose.pose_both_arm_2()
        elif idx == 9:
            armpose.pose_both_arm_3()


    def adjust_face(self, x_ang=0, y_ang=0, z_ang=0):
        # to move the bones, we need to go into pose mode
        bpy.ops.object.mode_set(mode='POSE')
        head = self.bones["head"] # manually selected skeleton
        head.rotation_mode = "XYZ"
        #print("transform: ", x_ang, -y_ang, z_ang)
        if np.abs(x_ang) > 1e-7:
            axis = 'X'
            head.rotation_euler.rotate_axis(axis, math.radians(x_ang))
        if np.abs(y_ang) > 1e-7:
            axis = 'Y'
            head.rotation_euler.rotate_axis(axis, math.radians(y_ang))
        if np.abs(z_ang) > 1e-7:
            axis = 'Z'
            head.rotation_euler.rotate_axis(axis, math.radians(z_ang))
        self.head_pose = (x_ang, y_ang, z_ang)
        return

    def adjust_neck(self, ang, axis="X"):
        bpy.ops.object.mode_set(mode='POSE')
        neck1 = self.bones["neck01"]
        neck1.rotation_mode = "XYZ"
        neck2 = self.bones["neck02"]
        neck2.rotation_mode = "XYZ"

        neck1.rotation_euler.rotate_axis(axis, math.radians(ang))
        neck2.rotation_euler.rotate_axis(axis, math.radians(ang))


    def set_neck(self, angle_scale=15, random_level=0.2):
        #Empirically moving both neck1 and neck2, looks more natural
        bpy.ops.object.mode_set(mode='POSE')
        neck2 = self.bones["neck02"]
        neck1 = self.bones["neck01"]
        neck2.rotation_mode = "XYZ"
        neck1.rotation_mode = "XYZ"

        # rotate X
        axis = 'X'
        angle = np.random.normal(0,random_level)*angle_scale
        neck1.rotation_euler.rotate_axis(axis, math.radians(angle))
        neck2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Y

        axis = 'Y'
        angle = np.random.normal(0,random_level)*angle_scale
        neck1.rotation_euler.rotate_axis(axis, math.radians(angle))
        neck2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Z
        axis = 'Z'
        angle = np.random.normal(0,random_level)*angle_scale
        neck1.rotation_euler.rotate_axis(axis, math.radians(angle))
        neck2.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

    def set_spine(self, angle_scale=15, random_level=0.15):
        bpy.ops.object.mode_set(mode='POSE')
        spine2 = self.bones["spine02"]
        spine1 = self.bones["spine01"]
        spine2.rotation_mode = "XYZ"
        spine1.rotation_mode = "XYZ"

        # rotate X
        axis = 'X'
        angle = np.random.normal(0,random_level)*angle_scale
        spine1.rotation_euler.rotate_axis(axis, math.radians(angle))
        spine2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Y

        axis = 'Y'
        angle = np.random.normal(0,random_level)*angle_scale
        spine1.rotation_euler.rotate_axis(axis, math.radians(angle))
        spine2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Z
        axis = 'Z'
        angle = np.random.normal(0,random_level)*angle_scale
        spine1.rotation_euler.rotate_axis(axis, math.radians(angle))
        spine2.rotation_euler.rotate_axis(axis, math.radians(angle))

    def random_transform(self):
        self.set_neck()
        self.set_spine()
        return

    def eye_closed(self):
        #eye status was marked in the file_name when generating MH models
        mark = self.object_name.split("_")[1]
        if mark == "o":
            return False
        return True

    def clean(self):
        clr_flag = True
        bpy.ops.wm.read_homefile()
        return


class IEEScenario(object):
    clr_flag = False

    def __init__(self, person, res_x, res_y, render_scale = 1.0):
        if IEEScenario.clr_flag:
            print("WARN: the scenario is not clear!")
        IEEScenario.clr_flag = False


        self.res_x = res_x
        self.res_y = res_y
        self.person = person

        self.nose_focus = person.get_nose_focus(shift=(0,0,-0.1)) #empirically
        self.forehead_focus = person.get_forehead_focus()
        self.breast_focus = person.get_breast_focus(shift=(0,-4.5,1.5)) #empirically

        self.scene = bpy.context.scene
        self.objectx = self.person.objectx

        self.scene.objects.active = self.objectx
        self.mat_world = self.objectx.matrix_world

        self.scene.render.resolution_x = res_x
        self.scene.render.resolution_y = res_y
        self.scene.render.resolution_percentage = 100

        self.render_size = (int(self.res_x * render_scale), int(self.res_y * render_scale))
        self.cam = self.scene.objects['Camera']
        #self.lamp = self.scene.objects['Lamp']
        self.lamps = {"Lamp":1}

        self.cam_looking_direction = None

        self.min_x = 1e7
        self.min_y = 1e7
        self.max_x = -1e7
        self.max_y = -1e7
        self.gaze = -1
        return


    # the random factors are introduced to generate faces in difference
    # location: where is the camera, default: self.breast_focus+ran_loc
    # target: where the camera look at, default: self.nose_focus+ran_tgt
    def set_camera(self, location=None, target=None, random_loc=0.8, random_tgt=0.8):
        ran_loc = (0,0,0)
        ran_tgt = (0,0,0)

        if random_loc:
            ran_loc = (np.random.rand(3)-0.5)*random_loc #an empirical setting

        if random_tgt:
            ran_tgt = (np.random.rand(3)-0.5)*random_tgt

        looking_direction = None

        if location:
            self.cam.location = location+random_loc

            if target:
                looking_direction = self.cam.location - mathutils.Vector(target+ran_tgt)
                rot_quat = looking_direction.to_track_quat('Z', 'Y')
                self.cam.rotation_euler = rot_quat.to_euler()

        else:
            self.cam.location = self.breast_focus+ran_loc
            if target:
                looking_direction = location - mathutils.Vector(target+ran_tgt)
                rot_quat = looking_direction.to_track_quat('Z', 'Y')
                self.cam.rotation_euler = rot_quat.to_euler()
            else:
                looking_direction = self.cam.location - mathutils.Vector(self.nose_focus+ran_tgt)
                rot_quat = looking_direction.to_track_quat('Z', 'Y')
                self.cam.rotation_euler = rot_quat.to_euler()

        self.cam_looking_direction = looking_direction

        return

    def get_lamp_params(self, name):
        if not name in self.lamps:
            return None
        lamp_obj = bpy.data.objects[name]
        lamp_data = lamp_obj.data
        lamp_cfg = {}
        lamp_cfg["location"] = lamp_obj.location
        lamp_cfg["type"] = lamp_data.type
        lamp_cfg["color"] = lamp_data.color
        lamp_cfg["energy"] = lamp_data.energy
        lamp_cfg["direct_xyz"] = (lamp_obj.rotation_euler.x, lamp_obj.rotation_euler.y, lamp_obj.rotation_euler.z)
        return lamp_cfg

    #if any manipulation, use the following commands
    # lamp_object.select = True
    # self.scene.objects.active = lamp_object
    def add_lamp(self, name, location, energy=1.0, direct_xyz=(0.873,-0.873,0.698), color=(1.0, 1.0, 1.0),type="POINT"):
        # if the lamp is already existed, nothing to add
        if name in self.lamps:
            return None
        lamp_data = bpy.data.lamps.new(name=name, type=type)
        lamp_data.energy = energy
        lamp_data.color = color
        lamp_object = bpy.data.objects.new(name=name, object_data=lamp_data)
        lamp_object.rotation_euler = Euler(direct_xyz, "XYZ")
        lamp_object.location = location
        self.scene.objects.link(lamp_object)
        self.lamps[name] = lamp_object
        return lamp_object

    # lamp_type: HEMI, POINT, SUN, SPOT, AREA
    def set_lamp(self, name="Lamp", energy=1.0, direct_xyz=(0.873,-0.873,0.698), color=(1.0, 1.0, 1.0), location=None,
                 random_leval=None, lamp_type="SUN"):
        # if the lamp name is not in the self.lamps, do nothing
        if not name in self.lamps:
            return None

        lamp_obj = bpy.data.objects[name]
        lamp_data = lamp_obj.data

        ran_loc = (0,0,0)
        if random_leval:
            ran_loc = (np.random.rand(3)-0.5)*random_leval

        if location:
            lamp_obj.location = location + ran_loc
        else:
            shift = np.array((0,-5,0.5)) + np.array(ran_loc) #empirically
            lamp_obj.location = self.nose_focus + shift

        lamp_obj.rotation_euler = Euler(direct_xyz, "XYZ")

        if lamp_type in {"HEMI", "POINT", "SUN", "SPOT", "AREA"}:
            lamp_data.type = lamp_type

        lamp_data.color = color

        lamp_data.energy = energy

        return

    def get_scenario_param(self):
        scenario = {}
        for ky in self.lamps:
            # lamp = bpy.data.lamps[ky]
            lamp_obj = bpy.data.objects[ky]
            lamp_data = lamp_obj.data
            sce_key = "lamp_loc_"+ky
            scenario[sce_key] = list(lamp_obj.location)
            sce_key = "lamp_type_"+ky
            scenario[sce_key] = lamp_data.type
            sce_key = "lamp_energy_"+ky
            scenario[sce_key] = lamp_data.energy
            sce_key = "lamp_color_"+ky
            scenario[sce_key] = list(lamp_data.color)
            sce_key = "lamp_direct_xyz_"+ky
            scenario[sce_key] = [lamp_obj.rotation_euler.x, lamp_obj.rotation_euler.y, lamp_obj.rotation_euler.z]
        scenario["cam_loc"] = list(self.cam.location)
        scenario["cam_look_direction"] = list(self.cam_looking_direction)
        scenario["head_pose"] = list(self.person.head_pose)
        return scenario

    def random_transform(self):
        self.set_camera()
        self.set_lamp()

    def setup(self, adj_neck=True, adj_deg=5):
        self.set_camera(random_loc=0, random_tgt=0)
        self.set_lamp(random_leval=0)
        if adj_neck:
            self.person.adjust_neck(adj_deg)

    def check_visibility(self, bvh, co_2d, tolerance = 0.004):
        pixel_x = -1
        pixel_y = -1
        if 0 <= co_2d.x <= 1.0 and 0 <= co_2d.y <= 1.0:
            # Try a ray cast, in order to test the vertex visibility from the camera
            location, normal, index, distance = bvh.ray_cast(self.cam.location, (ver.co - self.cam.location).normalized())
            # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
            if location and (ver.co - location).length < tolerance:
                pixel_x = round(co_2d[0] * self.render_size[0])
                pixel_y = round(co_2d[1] * self.render_size[1])
            else:
                pixel_x = -1
                pixel_y = -1
        return pixel_x, pixel_y


    def map_coord_from_3d_to_2d_face(self, label_data, coords_2d, chk_visi=True):
        #move outside
        bpy.context.scene.update()
        self.mat_world = self.objectx.matrix_world

        meshobj_name = label_data["meshname"]
        obj = bpy.data.objects[meshobj_name]
        obj.select = True
        bpy.context.scene.objects.active = obj
        bm_o = obj.to_mesh(self.scene, True, 'RENDER')
        bvh = None
        if chk_visi:
            bvh = self.create_bvh_tree(bm_o)

        #although label_data contains 3d coordinates,
        #manipulating the 3d model makes the coordinates unuseful...
        #TODO: remove the 3d coordinates from labels
        for ky in label_data:
            if not isinstance(ky, int):
                continue
            if ky > 68:  #69, 70 for eye center
                continue
            vid = list(label_data[ky].keys())[0]
            ver = bm_o.vertices[vid]
            ver.select = True
            co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*ver.co)
            ver.select = False

            pixel_x = round(co_2d[0] * self.render_size[0])
            pixel_y = round(co_2d[1] * self.render_size[1])
            if chk_visi and (not bvh):
                pixel_x, pixel_y = self.check_visibility(bvh, co_2d)

            coords_2d[ky] = [pixel_x, pixel_y]
        bpy.data.meshes.remove(bm_o)
        return

    def map_coord_from_3d_to_2d_eye(self, label_data, coords_2d, chk_visi=True):
        if not "eyename" in label_data:
            return
        bpy.context.scene.update()
        self.mat_world = self.objectx.matrix_world

        meshobj_name = label_data["eyename"]
        print("eye ball", meshobj_name)

        obj = bpy.data.objects[meshobj_name]
        obj.select = True
        bpy.context.scene.objects.active = obj
        bm_o = obj.to_mesh(self.scene, True, 'RENDER')
        bvh = None
        if chk_visi:
            bvh = self.create_bvh_tree(bm_o)

        #although label_data contains 3d coordinates,
        #manipulating the 3d model makes the coordinates unuseful...
        #TODO: remove the 3d coordinates from labels
        for ky in label_data:
            if not isinstance(ky, int):
                continue
            if ky < 69:  #69, 70 for eye center
                continue

            vid = list(label_data[ky].keys())[0]
            ver = bm_o.vertices[vid]
            ver.select = True
            co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*ver.co)
            pixel_x = round(co_2d[0] * self.render_size[0])
            pixel_y = round(co_2d[1] * self.render_size[1])
            if chk_visi and (not bvh):
                pixel_x, pixel_y = self.check_visibility(bvh, co_2d)

            coords_2d[ky] = [pixel_x, pixel_y]

        bpy.data.meshes.remove(bm_o)
        return

    def create_bvh_tree(self, bm_o):
        ##https://blender.stackexchange.com/questions/77607/how-to-get-the-3d-coordinates-of-the-visible-vertices-in-a-rendered-image-in-ble
        vertsInWorld = [self.mat_world * v.co for v in bm_o.vertices]
        bvh = BVHTree.FromPolygons( vertsInWorld, [p.vertices for p in bm_o.polygons] )
        return bvh


    def save_3dmodel_to_2dsample(self, label_data, file_name_prefix, chk_visi=True):
        # switch of log information
        #---------------------------------------------------------

        logfile = 'blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        #get mesh
        #bpy.context.scene.update()
        #self.mat_world = self.objectx.matrix_world

        coords_2d = {}

        self.map_coord_from_3d_to_2d_face(label_data, coords_2d, chk_visi)
        self.map_coord_from_3d_to_2d_eye(label_data, coords_2d,chk_visi)

        image_name = file_name_prefix+".png"
        label_name = file_name_prefix+".npy"

        data = {}
        data["config"] = self.get_scenario_param()
        data["label"] = coords_2d

        np.save(label_name, data)

        self.scene.render.filepath = image_name
        bpy.ops.render.render( write_still=True)

        os.close(1)
        os.dup(old)
        os.close(old)
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.select = True
            else:
                obj.select = False
        bpy.ops.object.delete()

        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)
        return


    def _min_max(self, co_2d):
        pixel_x = round(co_2d[0] * self.render_size[0])
        pixel_y = round(co_2d[1] * self.render_size[1])
        self.min_x = min(self.min_x, pixel_x)
        self.min_y = min(self.min_y, pixel_y)
        self.max_x = max(self.max_x, pixel_x)
        self.max_y = max(self.max_y, pixel_y)

    def _get_face_label(self):
        bpy.ops.object.mode_set(mode='POSE')

        #calculate face bound box
        #bones not counted in: speical06.R, special06.L, speical05.R, special05.L, head
        #mannully ...
        avoid_set = {"special06.R", "special06.L", "special05.R", "special05.L", "head", "jaw", "special04", "tongue00"}
        avoid_tail_set = {"eye.R", "eye.L", "orbicularis03.R", "orbicularis03.L", "orbicularis04.R", "orbicularis04.L"}

        for subbone in self.person.bones["head"].children_recursive:
            bone_name = subbone.name
            if bone_name in avoid_set:
                continue
            co_2d_h = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*subbone.head)
            self._min_max(co_2d_h)
            if bone_name in avoid_tail_set:
                continue
            co_2d_t = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*subbone.tail)
            self._min_max(co_2d_t)


        bpy.ops.object.mode_set(mode='OBJECT')
            
        if self.max_x < 10 or self.max_y < 10: #skip bad samples
            return (0,0,0,0)
        if self.min_x > self.res_x - 10 or self.min_y > self.res_y - 10:
            return (0,0,0,0)
        
        self.min_x = max(0, self.min_x)
        self.min_y = max(0, self.min_y)
        self.max_x = min(self.res_x, self.max_x)
        self.max_y = min(self.res_x, self.max_y)
        
        area = (self.max_y-self.min_y)*(self.max_x-self.min_x)
        if area < 100:
            return (0,0,0,0)
        return (self.min_x, self.res_y - self.min_y, self.max_x, self.res_y-self.max_y)
        
        
    def _get_gaze(self, render_scale=1.0):
        if self.person.eye_closed():
            return -1

        bpy.ops.object.mode_set(mode='POSE')
        eye_R = self.person.bones["eye.R"]
        eye_L = self.person.bones["eye.L"]

        rh_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_R.head)
        rt_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_R.tail)
        lh_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_L.head)
        lt_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_L.tail)
        bpy.ops.object.mode_set(mode='OBJECT')

        rh_pixel_x = round(rh_co_2d[0] * self.render_size[0])
        rh_pixel_y = round(rh_co_2d[1] * self.render_size[1])
        rt_pixel_x = round(rt_co_2d[0] * self.render_size[0])
        rt_pixel_y = round(rt_co_2d[1] * self.render_size[1])

        lh_pixel_x = round(lh_co_2d[0] * self.render_size[0])
        lh_pixel_y = round(lh_co_2d[1] * self.render_size[1])
        lt_pixel_x = round(lt_co_2d[0] * self.render_size[0])
        lt_pixel_y = round(lt_co_2d[1] * self.render_size[1])

        r_gaze = math.atan2(rt_pixel_y-rh_pixel_y,rt_pixel_x-rh_pixel_x)
        l_gaze = math.atan2(lt_pixel_y-lh_pixel_y,lt_pixel_x-lh_pixel_x)
        
        return round(math.degrees((r_gaze+l_gaze)/2.0))
        
    def save_to_sample(self, t_folder="./imgs/"):

        logfile = 'blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        bbox = self._get_face_label()
        gaze = self._get_gaze()
        s_label = str(bbox[0])+"_"+str(bbox[1])+"_"+str(bbox[2])+"_"+str(bbox[3])+"_"+str(gaze)
        fext = self.person.object_name.split(".")
        if not exists(t_folder):
                os.mkdir(t_folder)

        t_folder = t_folder+self.person.person_name+"/"
        if not exists(t_folder):
                os.mkdir(t_folder)

        filepath = t_folder+fext[0]+"_"+s_label+".png"
        self.scene.render.filepath = filepath
        bpy.ops.render.render( write_still=True)

        os.close(1)
        os.dup(old)
        os.close(old)
        return
                
        
        
    
    