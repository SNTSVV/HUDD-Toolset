"""
Author: Jun WANG @ IEE
"""

import numpy as np
from os.path import isfile, join, exists
import random
import os
from os import listdir
from ieeclass2 import IEEPerson, IEEScenario


def ensure_folder(folder_name):
    if not exists(folder_name):
        os.makedirs(folder_name)


def get_files(folder, extx=".csv"):
    all_files = []
    for f in listdir(folder):
        xf = join(folder, f)
        if isfile(xf) and xf.endswith(extx):
            all_files.append(xf)
    return all_files


class IEESimulator(object):
    def __init__(self, model_file, label_file, img_width, img_height):
        self.model_file = model_file
        self.label_file = label_file
        self.img_width = img_width
        self.img_height = img_height
        self.ensure_available()
        self.coods_3d = self.get_3d_coords_from_label()
        return

    # you can manipulate the 3D model here as you want
    def hook_ieeperson(self, person, cfg_to_person):
        x_ang, y_ang, z_ang = cfg_to_person
        person.adjust_face(x_ang, y_ang, z_ang)
        return

    # you can manipulate the blender envirionment here as you want
    def hook_ieescenario(self, scenario, cam_loc, cam_dir, lamp_loc):
        scenario.set_camera(location=cam_loc, target=cam_dir, random_loc=0, random_tgt=0)
        scenario.set_lamp(name="Lamp",
                          location=lamp_loc, random_leval=False, lamp_type="SUN")
        return

    def get_3d_coords_from_label(self):
        coods_3d = np.load(self.label_file, allow_pickle=True)
        coods_3d = coods_3d.item()
        return coods_3d

    def customize_sample(self, cfg_to_person, cam_loc, cam_dir, lamp_loc, dst=None):
        person = IEEPerson(self.model_file)
        # print("person = IEEPerson(self.model_file)")

        self.hook_ieeperson(person, cfg_to_person)
        scenario = IEEScenario(person, res_x=self.img_width, res_y=self.img_height)
        self.hook_ieescenario(scenario, cam_loc, cam_dir, lamp_loc)

        if dst:
            label_data = self.get_3d_coords_from_label()
            scenario.save_3dmodel_to_2dsample(self.coods_3d, dst)
            person.clean()

    def ensure_available(self):
        vtrue = True
        if not isfile(self.model_file):
            print("cannot find pose file: ", self.model_file)
            vtrue = False

        if not isfile(self.label_file):
            print("cannot find pose file: ", self.label_file)
            vtrue = False

        assert vtrue
        assert self.img_width > 1 and self.img_height > 1


class IEEImitator(IEESimulator):
    def __init__(self, model_file, label_file, pose_file, img_width, img_height, dst_datafolder, sampling):
        super(IEEImitator, self).__init__(model_file, label_file, img_width, img_height)
        self.pose_file = pose_file
        assert isfile(self.pose_file)
        self.sampling = sampling
        self.dst, self.pose_who, self.model_who = self.create_dst_datafolder(dst_datafolder)

    def get_pose_data(self):
        pose_data = np.loadtxt(self.pose_file, delimiter=',', skiprows=1)
        pose_data = pose_data[:, [0, 10, 11, 12]].astype(np.float32)
        if self.sampling:
            # using only even rows pose data
            pose_data = pose_data[::2]
        # print("pose_data: ", pose_data.shape)
        return pose_data

    def create_dst_datafolder(self, dst_datafolder):
        pose_who = self.pose_file.split("/")[-1]
        pose_who = pose_who.split(".")[0]
        model_who = self.model_file.split("/")[-1]
        model_who = model_who.split(".")[0]

        if not dst_datafolder:
            dst_datafolder = "./iee_imgs/"
        # dst = dst_datafolder+model_who
        dst = dst_datafolder
        ensure_folder(dst)
        return dst, pose_who, model_who

    def discovery_used_poses(self):
        all_png_files = get_files(self.dst, ".png")
        all_idxs = []
        for afile in all_png_files:
            tmp_str = afile.split(".png")[0]
            identifier = tmp_str.split("/")[-1]
            all_idxs.append(identifier)
        return set(all_idxs)

    # you can manipulate the 3D model here as you want
    def hook_ieeperson(self, person, cfg_to_person):
        x_ang, y_ang, z_ang = cfg_to_person
        person.adjust_face(x_ang, y_ang, z_ang)
        return

    # you can manipulate the blender envirionment here as you want
    def hook_ieescenario(self, scenario, cam_loc, cam_dir, lamp_loc):
        scenario.setup()
        return

    def create_samples_allParams(self, cam_dir, cam_loc, lamp_loc, head, num, imgPath):
        pose_data = self.get_pose_data()
        used_poses = self.discovery_used_poses()
        counter = 0

        while counter < num:
            person = IEEPerson(self.model_file)
            row = pose_data[random.randint(0, len(pose_data) - 1)]
            frame = int(row[0])
            identifier = self.pose_who + "_" + str(frame)
            if identifier in used_poses:
                continue
            file_name_prefix = imgPath
            cfg_to_person = (row[3], -row[2], row[1])
            cfg_to_person = (head[0], head[1], head[2])
            self.hook_ieeperson(person, cfg_to_person)
            scenario = IEEScenario(person, res_x=self.img_width, res_y=self.img_height)
            cfg_to_camloc = (cam_loc[0], cam_loc[1], cam_loc[2])
            cfg_to_camdir = (cam_dir[0], cam_dir[1], cam_dir[2])
            cfg_to_lamploc = (lamp_loc[0], lamp_loc[1], lamp_loc[2])
            self.hook_ieescenario(scenario, cfg_to_camloc, cfg_to_camdir, cfg_to_lamploc)
            if file_name_prefix:
                label_data = self.get_3d_coords_from_label()
                scenario.save_3dmodel_to_2dsample(self.coods_3d, file_name_prefix)
                person.clean()
            # self.customize_sample(cfg_to_person, cfg_to_camloc, cfg_to_camdir, cfg_to_lampeng, cfg_to_lampdir,
            # cfg_to_lampcol, cfg_to_lamploc, dst=file_name_prefix)
            counter += 1
        # print("generate img: ---", file_name_prefix, "---")

    def create_samples_chosenParams(self, head, lamp_loc, cam_loc, cam_dir, num):
        pose_data = self.get_pose_data()
        used_poses = self.discovery_used_poses()
        counter = 0

        while counter < num:
            person = IEEPerson(self.model_file)
            row = pose_data[random.randint(0, len(pose_data) - 1)]
            frame = int(row[0])
            identifier = self.pose_who + "_" + str(frame)
            if identifier in used_poses:
                continue
            file_name_prefix = self.dst + "/" + self.pose_who + "_" + str(int(frame))
            cfg_to_person = (row[3], -row[2], row[1])
            head_x = head[0]
            head_y = head[2]
            head_z = head[4]
            lamp_loc_x = lamp_loc[0]
            lamp_loc_y = lamp_loc[2]
            lamp_loc_z = lamp_loc[4]
            cam_loc_x = cam_loc[0]
            cam_loc_y = cam_loc[2]
            cam_loc_z = cam_loc[4]
            cam_dir_x = cam_dir[0]
            cam_dir_y = cam_dir[2]
            cam_dir_z = cam_dir[4]
            head_valx = random.uniform(head[1][0], head[1][1])
            head_valy = random.uniform(head[3][0], head[3][1])
            head_valz = random.uniform(head[5][0], head[5][1])
            lamp_loc_valx = random.uniform(lamp_loc[1][0], lamp_loc[1][1])
            lamp_loc_valy = random.uniform(lamp_loc[3][0], lamp_loc[3][1])
            lamp_loc_valz = random.uniform(lamp_loc[5][0], lamp_loc[5][1])
            cam_loc_valx = random.uniform(cam_loc[1][0], cam_loc[1][1])
            cam_loc_valy = random.uniform(cam_loc[3][0], cam_loc[3][1])
            cam_loc_valz = random.uniform(cam_loc[5][0], cam_loc[5][1])
            cam_dir_valx = random.uniform(cam_dir[1][0], cam_dir[1][1])
            cam_dir_valy = random.uniform(cam_dir[3][0], cam_dir[3][1])
            cam_dir_valz = random.uniform(cam_dir[5][0], cam_dir[5][1])
            # print(file_name_prefix)

            if head_x:
                cfg_to_person = (head_valx, -row[2], row[1])
            if head_y:
                cfg_to_person = (row[3], head_valy, row[1])
            if head_z:
                cfg_to_person = (row[3], -row[2], head_valz)
            if head_x and head_y:
                cfg_to_person = (head_valx, head_valy, row[1])
            if head_x and head_z:
                cfg_to_person = (head_valx, -row[2], head_valz)
            if head_y and head_z:
                cfg_to_person = (row[3], head_valy, head_valz)
            if head_x and head_y and head_z:
                cfg_to_person = (head_valx, head_valy, head_valz)

            self.hook_ieeperson(person, cfg_to_person)
            scenario = IEEScenario(person, res_x=self.img_width, res_y=self.img_height)
            ran_cam_loc = scenario.breast_focus + ((np.random.rand(3) - 0.5) * 0.8)  # an empirical setting
            cfg_to_camloc = ran_cam_loc

            if cam_loc_x:
                cfg_to_camloc = (cam_loc_valx, ran_cam_loc[1], ran_cam_loc[2])
            if cam_loc_y:
                cfg_to_camloc = (ran_cam_loc[0], cam_loc_valy, ran_cam_loc[2])
            if cam_loc_z:
                cfg_to_camloc = (ran_cam_loc[0], ran_cam_loc[1], cam_loc_valz)
            if cam_loc_x and cam_loc_y:
                cfg_to_camloc = (cam_loc_valx, cam_loc_valy, ran_cam_loc[2])
            if cam_loc_x and cam_loc_z:
                cfg_to_camloc = (cam_loc_valx, ran_cam_loc[1], cam_loc_valz)
            if cam_loc_y and cam_loc_z:
                cfg_to_camloc = (ran_cam_loc[0], cam_loc_valy, cam_loc_valz)
            if cam_loc_x and cam_loc_y and cam_loc_z:
                cfg_to_camloc = (cam_loc_valx, cam_loc_valy, cam_loc_valz)

            ran_cam_dir = scenario.nose_focus + ((np.random.rand(3) - 0.5) * 0.8)
            cfg_to_camdir = ran_cam_dir

            if cam_dir_x:
                cfg_to_camdir = (cam_dir_valx, ran_cam_dir[1], ran_cam_dir[2])
            if cam_dir_y:
                cfg_to_camdir = (ran_cam_dir[0], cam_dir_valy, ran_cam_dir[2])
            if cam_dir_z:
                cfg_to_camdir = (ran_cam_dir[0], ran_cam_dir[1], cam_dir_valz)
            if cam_dir_x and cam_dir_y:
                cfg_to_camdir = (cam_dir_valx, cam_dir_valy, ran_cam_dir[2])
            if cam_dir_x and cam_dir_z:
                cfg_to_camdir = (cam_dir_valx, ran_cam_dir[1], cam_dir_valz)
            if cam_dir_y and cam_dir_z:
                cfg_to_camdir = (ran_cam_dir[0], cam_dir_valy, cam_dir_valz)
            if cam_dir_x and cam_dir_y and cam_dir_z:
                cfg_to_camdir = (cam_dir_valx, cam_dir_valy, cam_dir_valz)

            ran_lamp_dir = (random.uniform(0.29, 0.87), random.uniform(-5.0, -0.87), random.uniform(0.69, 14.70))
            cfg_to_lampdir = ran_lamp_dir

            ran_lamp_col = (random.uniform(-4.29, 1.0), random.uniform(-1.39, 1.0), 1.0)
            cfg_to_lampcol = ran_lamp_col

            shift = np.array((0, -5, 0.5)) + np.array(np.random.rand(3) - 0.5)
            ran_lamp_loc = scenario.nose_focus + shift
            cfg_to_lamploc = ran_lamp_loc

            if lamp_loc_x:
                cfg_to_lamploc = (lamp_loc_valx, ran_lamp_loc[1], ran_lamp_loc[2])
            if lamp_loc_y:
                cfg_to_lamploc = (ran_lamp_loc[0], lamp_loc_valy, ran_lamp_loc[2])
            if lamp_loc_z:
                cfg_to_lamploc = (ran_lamp_loc[0], ran_lamp_loc[1], lamp_loc_valz)
            if lamp_loc_x and lamp_loc_y:
                cfg_to_lamploc = (lamp_loc_valx, lamp_loc_valy, ran_lamp_loc[2])
            if lamp_loc_x and lamp_loc_z:
                cfg_to_lamploc = (lamp_loc_valx, ran_lamp_loc[1], lamp_loc_valz)
            if lamp_loc_y and lamp_loc_z:
                cfg_to_lamploc = (ran_lamp_loc[0], lamp_loc_valy, lamp_loc_valz)
            if lamp_loc_x and lamp_loc_y and lamp_loc_z:
                cfg_to_lamploc = (lamp_loc_valx, lamp_loc_valy, lamp_loc_valz)

            ran_lamp_eng = random.uniform(-0.08, 1.0)
            cfg_to_lampeng = ran_lamp_eng
            self.hook_ieescenario(scenario, cfg_to_camloc, cfg_to_camdir, cfg_to_lampeng)

            if file_name_prefix:
                label_data = self.get_3d_coords_from_label()
                scenario.save_3dmodel_to_2dsample(self.coods_3d, file_name_prefix)
                person.clean()
            # self.customize_sample(cfg_to_person, cfg_to_camloc, cfg_to_camdir, cfg_to_lampeng, cfg_to_lampdir, cfg_to_lampcol, cfg_to_lamploc, dst=file_name_prefix)
            counter += 1
        # print("generate img: ---", file_name_prefix, "---")
