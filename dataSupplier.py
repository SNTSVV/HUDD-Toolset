#
# Copyright (c) IEE 2019-2020.
# Created by Jun WANG, jun.wang@iee.lu, IEE, 2019.
# Modified by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
#
import subprocess as sp
import pathlib as pl
import dnnModels
import HeatmapModule
import testModule
#from searchModule import setX, setNewX, doImage
#import searchModule
#from assignModule import testModule, HeatmapModule
from imports import shutil, random, np, pd, math, glob, json, dlib, cv2, os, torch, Image, Variable, datasets, \
    transforms, DataLoader, Dataset, SubsetRandomSampler, setupTransformer, normalize, isfile, join, exists, basename, \
    dirname, makedirs, rmtree, subprocess, time
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import config as cfg

components = cfg.components
blenderPath = cfg.blenderPath
nVar = cfg.nVar

globalCounter = random.randint(1, 999999999)
import scipy.misc as sc
import random
import csv
import imageio
# components = ["mouth", "noseridge", "nose", "rightbrow", "righteye", "lefteye", "leftbrow"]
components = ["mouth"]
outputPath = "/Users/hazem.fahmy/Documents/HPD/"
#outputPath = "/home/users/hfahmy/DEEP/HPC/HPD/"
outputPath = "/Users/android/Documents/HPD/"
DIR = "/Users/hazem.fahmy/Documents/HPC/HUDD/runPy/"
#DIR = "/home/users/hfahmy/DEEP/HPC/HUDD/runPy/"
DIR = "/Users/android/Documents/HPC/HUDD/runPy/"
path = join(outputPath, "IEEPackage")
#import pandas as pd
import numpy as np
train_max_num = 8192+4096
test_max_num = 1024+512
real_max_num = 1024+512

total_epoch = 100
best_model_path = "./bst_model/kpmodel.pt"
loss_file_path = "./bst_model/loss.npy"
plot_results = "./results_kaggle"
labels = ["lefteyebrow", "righteyebrow", "lefteye", "righteye", "nose", "mouth"]
target_size = 128
iee_img_width = 376
iee_img_height = 240
cood_num = 27 #FIXME
#cood_num = 36 #FIXME
h_tol = 25
w_tol = 25
validRatio = 0.1
pinMemory = True
width = 128
height = 128
sigma = 5
gaussian_scale = 10.0
n_points = 64 #FIXME
batch_size = 64
data_random_seed = 3
gpu_id = 1
iee_train_data = "./dataset/ieetrain.npy"
iee_test_data = "./dataset/ieetest.npy"
iee_real_data = "./dataset/ieereal.npy"

FILE_EXT = '.jpg'
PATH = './train_model/'

class DataReader(object):
    def __init__(self, data_dir=None, file_ext=FILE_EXT, sequential=False, mode=3):
        self.data_dir = data_dir
        if mode == 1:
            self.load_all()
        elif mode == 2:
            self.load_R()
        elif mode == 3:
            self.load_Sim()
    def load_all(self):
        xs = []
        xs2 = []
        ys = []
        ys2 = []
        fid = []
        fid2 = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        total = 0
        reader = os.listdir(join(self.data_dir, "TrainingSet_BNG"))
        for row in reader:
            angle = float(os.path.basename(row).split(".jpg")[0])
            #print(os.path.basename(row).split(".")[0], angle)
            xs.append(os.path.join(self.data_dir, row))
            ys.append(angle)
            fid.append(total)
            total += 1


        xs2, ys2, fid2 = self.loadCSV(self.data_dir + '/steering5.csv', 'angle', 'timestamp', 'hm5')
        self.num_images = len(xs)
        print("Total Images", self.num_images)
        print("Total Images", len(xs2))
        xsf_train = [x for x in xs[:int(len(xs) * 0.8)]]
        ysf_train = [x for x in ys[:int(len(xs) * 0.8)]]
        fid_train = [x for x in fid[:int(len(xs) * 0.8)]]
        for x in xs2[:int(len(xs2) * 0.8)]:
            xsf_train.append(x)
        for y in ys2[:int(len(xs2) * 0.8)]:
            ysf_train.append(y)
        for f in fid2[:int(len(xs2) * 0.8)]:
            fid_train.append(f)
        print("Train", len(xsf_train))
        xsf_test = [x for x in xs[-int(len(xs) * 0.2):]]
        ysf_test = [x for x in ys[-int(len(xs) * 0.2):]]
        fid_test = [x for x in fid[-int(len(xs) * 0.2):]]
        for x in xs2[-int(len(xs2) * 0.2):]:
            xsf_test.append(x)
        for y in ys2[-int(len(xs2) * 0.2):]:
            ysf_test.append(y)
        for f in fid2[-int(len(xs2) * 0.2):]:
            fid_test.append(f)

        print("Test", len(xsf_test))

        self.train_xs = xsf_train
        self.train_ys = ysf_train
        self.train_fid = fid_train
        self.val_xs = xsf_test
        self.val_ys = ysf_test
        self.val_fid = fid_test

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)
        print('Train data:', self.num_train_images)
        print('Test data:', self.num_val_images)


        return xs, ys, fid

    def load(self):
        xs = []
        ys = []
        fid = []
        xs2 = []
        ys2 = []
        fid2 = []
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        total = 0
        count01 = count005 = count002 = count0 = 0

        #with open('data.csv') as f:
        #    reader = csv.DictReader(f)
        reader = os.listdir(self.data_dir)
        for row in reader:
            angle = float(os.path.basename(row).split(".jpg")[0])
            #print(os.path.basename(row).split(".")[0], angle)
            xs.append(os.path.join(self.data_dir, row))
            ys.append(angle)
            fid.append(total)
            total += 1

        reader = os.listdir(self.data_dir)
        f = open(PATH + 'testdata.txt', 'r')
        lines = f.readlines()
        i = 0
        for i in range(0, len(lines)):
            line = lines[i]
            n = line.split(",")[0]
            n1 = line.split(",")[1]

            angle = float(n1)
            #print(os.path.basename(row).split(".")[0], angle)
            xs2.append(os.path.join(PATH, 'TrainingSet', n + ".jpg"))
            ys2.append(angle)
            fid2.append(total)
            total += 1
        #with open('train_center.csv') as f:
        #    reader = csv.DictReader(f)
        #    for row in reader:
        #        angle = float(row['steering_angle'])
        #        xs.append(DATA_DIR + 'Ch2_Train/center/' + row['frame_id'] + FILE_EXT)
        #        ys.append(row['steering_angle'])
        #        total += 1

        print('> 0.1 or < -0.1: ' + str(count01))
        print('> 0.05 or < -0.05: ' + str(count005))
        print('> 0.02 or < -0.02: ' + str(count002))
        print('~0: ' + str(count0))
        print('Total data: ' + str(total))

        self.num_images = len(xs)

        c = list(zip(xs, ys))
        c2 = list(zip(xs2, ys2))
        random.shuffle(c)
        random.shuffle(c2)
        xs, ys = zip(*c)
        xs2, ys2 = zip(*c2)

        self.train_xs = xs
        self.train_ys = ys
        self.train_fid = fid
        self.val_xs = xs2
        self.val_ys = ys2
        self.val_fid = fid2

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)
        print('Train data:', self.num_train_images)
        print('Test data:', self.num_val_images)

    def load_R(self):
        xs = []
        ys = []
        fid = []
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        print(self.data_dir)
        #xs1, ys1, fid1 = self.loadCSV(self.data_dir + '/steering1.csv', 'angle', 'timestamp', 'hm1')
        xs2, ys2, fid2 = self.loadCSV(self.data_dir + '/steering2.csv', 'angle', 'timestamp', 'hm2')
        xs3, ys3, fid3 = self.loadCSV(self.data_dir + '/steering4.csv', 'angle', 'timestamp', 'hm4')
        #xs4, ys4, fid4 = self.loadCSV(self.data_dir + '/steering5.csv', 'angle', 'timestamp', 'hm5')
        #xs5, ys5, fid5 = self.loadCSV(self.data_dir + '/steering5.csv', 'angle', 'timestamp')
        #xs6, ys6, fid6 = self.loadCSV(self.data_dir + '/steering6.csv', 'angle', 'timestamp')
        #for x in xs1:
        #    xs.append(x)
        #for y in ys1:
        #    ys.append(y)
        #for id in fid1:
        #    fid.append(id)
        for x in xs2:
            xs.append(x)
        for y in ys2:
            ys.append(y)
        for id in fid2:
            fid.append(id)
        for x in xs3:
            xs.append(x)
        for y in ys3:
            ys.append(y)
        for id in fid3:
            fid.append(id)
        #for x in xs4:
        #    xs.append(x)
        #for y in ys4:
        #    ys.append(y)
        #for id in fid4:
        #    fid.append(id)
        #for x in xs5:
        #    xs.append(x)
        #for y in ys5:
        #    ys.append(y)
        #for id in fid5:
        #    fid.append(id)
        #for x in xs6:
        #    xs.append(x)
        #for y in ys6:
        #    ys.append(y)
        #for id in fid6:
        #    fid.append(id)

        self.num_images = len(xs)

        #c = list(zip(xs, ys))
        #random.shuffle(c)
        #xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(xs) * 0.8)]
        self.train_fid = fid[:int(len(xs)* 0.8)]
        self.val_xs = xs[-int(len(xs) * 0.2):]
        self.val_ys = ys[-int(len(xs) * 0.2):]
        self.val_fid = fid[-int(len(xs) * 0.2):]

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)
        print('Train data:', self.num_train_images)
        print('Test data:', self.num_val_images)
    def load_Sim(self):

        xs = []
        xs2 = []
        ys = []
        ys2 = []
        fid = []
        fid2 = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        self.sim_batch_pointer = 0
        total = 0
        path = join(self.data_dir, "TrainingSet_BNG")
        reader = os.listdir(path)
        for row in reader:
            angle = float(os.path.basename(row).split(".jpg")[0])
            #print(os.path.basename(row).split(".")[0], angle)
            xs.append(os.path.join(path, os.path.basename(row)))
            ys.append(angle)
            fid.append(total)
            total += 1

        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)
        self.num_images = len(xs)
        print("Total Simulator Images ", self.num_images)
        xsf_train = [x for x in xs[:int(len(xs) * 0.8)]]
        ysf_train = [x for x in ys[:int(len(xs) * 0.8)]]
        fid_train = [x for x in fid[:int(len(xs) * 0.8)]]

        xsf_test, ysf_test, fid_test = self.loadCSV(self.data_dir + '/steering5.csv', 'angle', 'timestamp', 'hm5')

        for x in xsf_test[-int(len(xsf_test) * 0.2):]:
            xsf_train.append(x)
        for y in ysf_test[-int(len(xsf_test) * 0.2):]:
            ysf_train.append(y)
        for id in fid_test[-int(len(xsf_test) * 0.2):]:
            fid_train.append(id)

        print("Train Simulator Images", len(xsf_train))
        xsf_test_sim = [x for x in xs[-int(len(xs) * 0.2):]]
        ysf_test_sim = [x for x in ys[-int(len(xs) * 0.2):]]
        fid_test_sim = [x for x in fid[-int(len(xs) * 0.2):]]
        print("Test Simulator Images", len(xsf_test_sim))


        print("Real Test Images", len(xsf_test))

        self.train_xs = xsf_train
        self.train_ys = ysf_train
        self.train_fid = fid_train
        self.val_xs = xsf_test[:int(len(xsf_test) * 0.2)]
        self.val_ys = ysf_test[:int(len(xsf_test) * 0.2)]
        self.val_fid = fid_test[:int(len(xsf_test) * 0.2)]
        self.sim_xs = xsf_test_sim
        self.sim_ys = ysf_test_sim
        self.sim_fid = fid_test_sim

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)
        self.num_sim_images = len(self.sim_xs)
        print('Train Simulator data:', self.num_train_images)
        print('Test Real data:', self.num_val_images)
        print('Test Sim data:', self.num_sim_images)


        return xs, ys, fid
    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        fid_out = []
        for i in range(0, batch_size):
            image = imageio.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            x_out.append(np.array(Image.fromarray(image[-400:]).resize([200, 66])) / 255.0)
            fid_out.append([self.train_fid[(self.train_batch_pointer + i) % self.num_train_images]])
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out, fid_out

    def load_val_batch(self, batch_size):
        x_out = []
        y_out = []
        fid_out = []
        for i in range(0, batch_size):
            image = imageio.imread(self.val_xs[(self.val_batch_pointer + i) % self.num_val_images])
            fid_out.append([self.val_fid[(self.val_batch_pointer + i) % self.num_val_images]])
            x_out.append(np.array(Image.fromarray(image[-400:]).resize([200, 66])) / 255.0)
            y_out.append([self.val_ys[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return x_out, y_out, fid_out


    def load_sim_batch(self, batch_size):
        x_out = []
        y_out = []
        fid_out = []
        for i in range(0, batch_size):
            image = imageio.imread(self.sim_xs[(self.sim_batch_pointer + i) % self.num_sim_images])
            fid_out.append([self.sim_fid[(self.sim_batch_pointer + i) % self.num_sim_images]])
            x_out.append(np.array(Image.fromarray(image[-400:]).resize([200, 66])) / 255.0)
            y_out.append([self.sim_ys[(self.sim_batch_pointer + i) % self.num_sim_images]])
        self.sim_batch_pointer += batch_size
        return x_out, y_out, fid_out

    def load_seq(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        print('LSTM Data')

        with open('train_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(DATA_DIR + 'Ch2_Train/center/' + row['frame_id'] + FILE_EXT)
                ys.append(row['steering_angle'])

        c = list(zip(xs, ys))
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 1.0)]
        self.train_ys = ys[:int(len(xs) * 1.0)]

        self.num_images = len(self.train_xs)
        print('total: ' + str(self.num_images))

        self.num_train_images = len(self.train_xs)
    def loadCSV(self, csvfile, key_sa, key_fid, dataName):
        xs = []
        ys = []
        fid = []
        fid_all = []
        total = 0
        count01 = count005 = count002 = count0 = 0
        files = os.listdir(join(self.data_dir, dataName))
        #for file in files:
        #    fid_all.append(int(os.path.basename(file).split(".jpg")[0]))
        with open(csvfile) as f:
            reader = csv.DictReader(f)
            for row in reader:
                angle = float(row[key_sa])
                #print(row[key_fid])
                #absolute_difference_function = lambda list_value: abs(list_value - int(row[key_fid]))
                #closest_value = min(fid_all, key=absolute_difference_function)
                #print(row[key_fid], closest_value)
                #print(row[key_sa])
                #print(row[key_fid])
                filepath = join(self.data_dir, dataName, str(int(row[key_fid]))) + FILE_EXT
                #filepath = join(self.data_dir, dataName, str(int(closest_value))) + FILE_EXT

                #filepath2 = join(self.data_dir, "left", str(int(row[key_fid]))) + FILE_EXT
                #print(filepath)

                if isfile(filepath):
                    xs.append(filepath)
                    ys.append(float(row[key_sa]))
                    fid.append(int(row[key_fid]))
                #else:
                #    print(row[key_fid], closest_value, angle)
                total += 1
            print(total, " images in folder")

        #with open('train_center.csv') as f:
        #    reader = csv.DictReader(f)
        #    for row in reader:
        #        angle = float(row['steering_angle'])
        #        xs.append(DATA_DIR + 'Ch2_Train/center/' + row['frame_id'] + FILE_EXT)
        #        ys.append(row['steering_angle'])
        #        total += 1

        print('> 0.1 or < -0.1: ' + str(count01))
        print('> 0.05 or < -0.05: ' + str(count005))
        print('> 0.02 or < -0.02: ' + str(count002))
        print('~0: ' + str(count0))
        print('Total data: ' + str(total))
        return xs, ys, fid
    def load_seq_2(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        print('LSTM Data')

        with open('interpolated_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(DATA_DIR + 'output/center/' + row['frame_id'] + FILE_EXT)
                ys.append(row['steering_angle'])

        c = list(zip(xs, ys))
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 1.0)]
        self.train_ys = ys[:int(len(xs) * 1.0)]

        self.num_images = len(self.train_xs)
        print('total: ' + str(self.num_images))

        self.num_train_images = len(self.train_xs)

    def skip(self, num):
        self.train_batch_pointer += num

    def load_seq_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = sc.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            x_out.append(np.array(Image.fromarray(image[-400:]).resize([200, 66])) / 255.0)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out

def processX(x):
    out = str(x[0])[0:5]
    for i in range(1, len(x)):
        out += "_" + str(x[i])[0:5]
    return out

def getPosePath():
    model_folder = "mhx2"
    label_folder = "newlabel3d"
    pose_folder = "pose"
    model_file = ["Aac01_o", "Aaj01_o", "Aai01_c", "Aah01_o", "Aaf01_o", "Aag01_o", "Aab01_o", "Aaa01_o", "Aad01_o"]  # TrainingSet
    # model_file = [model_folder + "/Aae01_o", model_folder + "/Aaa01_o"] #ImprovementSet1
    # model_file = [model_folder + "/Aae01_o"] #ImprovementSet2
    # model_file = ["Aad01_o"]  # TestSet
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aae01_o"] #TestSet1
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aah01_o"] #TestSet2
    label_file = ["aac01_o", "aaj01_o", "aai01_c", "aah01_o", "aaf01_o", "aag01_o", "aab01_o", "aaa01_o", "aad01_o"]  # TrainingSet
    # label_file = [label_folder + "/aae01_o", label_folder + "/aaa01_o"] #ImprovementSet1
    # label_file = [label_folder + "/aae01_o"] #ImprovementSet2
    # label_file = ["aad01_o"]  # TestSet
    # label_file = [label_folder + "/aad01_o", label_folder + "/aae01_o"] #TestSet1
    # label_file = [label_folder + "/aad01_o", label_folder + "/aah01_o"] #TestSet2
    pose_file = ["Aga", "Pbo", "Mbg", "Ldh", "Gle", "Hcl", "Acr", "Acq", "Dho"]  # TrainingSet
    # pose_file = [pose_folder + "/Fga", pose_folder + "/Acq"] #ImprovementSet1
    # pose_file = [pose_folder + "/Fga"] #ImprovementSet2
    # pose_file = ["Dho"]  # TestSet
    # pose_file = [pose_folder + "/Dho", pose_folder + "/Fga"] #TestSet1
    # pose_file = [pose_folder + "/Dho", pose_folder + "/Ldh"] #TestSet2
    return model_folder, label_folder, pose_folder, model_file, label_file, pose_file



def generateHuman(x, caseFile):
    # x = [head0/1/2, lamp_col0/1/2, lamp_loc0/1/2, lamp_dir0/1/2, cam2, age, eye_hue, eye_iris, eye_sat,
    # eye_val, skin_freckle, skin_oil, skin_veins, eye_color, gender]
    # x[0:12] -- Scenario/Person
    # x[13:23] -- MBLab/Human
    global globalCounter
    #x[19] = math.floor(x[19])
    #print(x)
    SimDataPath = caseFile["SimDataPath"]
    imgPath = join(SimDataPath, str(globalCounter), processX(x))
    t1 = time.time()
    if not isfile(join(SimDataPath, processX(x)) + ".png"):
        # output = os.path.join(pl.Path(__file__).resolve().parent.parent.parent, 'snt_simulator', 'data', 'mblab')
        script = os.path.join(caseFile["outputPath"],
                              "IEEPackage\mblab-interface\scripts\snt_face_dataset_generation.py")
        data_path = join(caseFile["outputPath"], "IEEPackage\mblab_asset_data")
        cmd = [str(blenderPath), "-b", "--log-level", str(0), "-noaudio", "--python", script, "--", str(data_path), "-l", "debug", "-o",
               f"{imgPath}", "--render", "--studio-lights"]  # generate MBLab character
        try:
            devnull = open(os.devnull, 'w')
            sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
        except Exception as e:
            print("Error in MBLab creating")
            print(e)
            #exit(0)

        filePath = os.path.join(pl.Path(__file__).parent.resolve(), "ieekeypoints2.py")
        try:
            devnull = open(os.devnull, 'w')
            sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            cmd = [str(blenderPath), "--background", "-noaudio", "--verbose", str(0), "--python", str(filePath), "--",
                 "--imgPath", str(imgPath)]
            sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            # str(imgPath)], stdout=subprocess.PIPE)
            shutil.copy(imgPath + ".png", join(SimDataPath, processX(x) + ".png"))
            shutil.copy(imgPath + ".npy", join(SimDataPath, processX(x) + ".npy"))
            shutil.rmtree(join(SimDataPath, str(globalCounter)))
        except Exception as e:
            print("error in Blender scenario creation")
            print(e)
            #exit(0)
        # print(ls)
        globalCounter += 1
    img = cv2.imread(join(SimDataPath, processX(x)) + ".png")
    t2 = time.time()
    print("Image Generation: ", str(t2-t1)[0:5], end="\r")
    if img is None:
        print("image not found, not processed")
        #exit(0)
        return imgPath, False
    if len(img) > 128:
        faceFound = processImage(join(SimDataPath, processX(x)) + ".png", join(caseFile["outputPath"],
                                                        "IEEPackage/clsdata/mmod_human_face_detector.dat"))
    else:
        faceFound = True

    #shutil.copy(join(SimDataPath, processX(x)) + ".png", join(caseFile["GI"], str(caseFile["ID"]), "images", processX(x) + ".png"))
    return join(SimDataPath, processX(x)), faceFound


def generateAnImage(x, caseFile):
    SimDataPath = caseFile["SimDataPath"]
    outPath = caseFile["outputPath"]
    cam_dir = [False, (x[0], x[0]), False, (x[1], x[1]), False, (x[2], x[2])]
    cam_loc = [False, (x[3], x[3]), False, (x[4], x[4]), False, (x[5], x[5])]
    lamp_loc = [False, (x[6], x[6]), False, (x[7], x[7]), False, (x[8], x[8])]
    lamp_col = [False, (x[9], x[9]), False, (x[10], x[10]), False, (x[11], x[11])]
    lamp_dir = [False, (x[12], x[12]), False, (x[13], x[13]), False, (x[14], x[14])]
    lamp_eng = [False, (x[15], x[15])]
    head = [False, (x[16], x[16]), False, (x[17], x[17]), False, (x[18], x[18])]
    model_folder, label_folder, pose_folder, model_file, label_file, pose_file = getPosePath()
    m = random.randint(0, len(pose_file) - 1)
    m = int(math.floor(x[19]))
    imgPath = join(SimDataPath, processX(x))
    blenderPath = "/Applications/Blender.app/Contents/MacOS/blender"
    #blenderPath = "/home/users/hfahmy/blender-2.79/blender"
    filePath = DIR+"ieekeypoints.py"
    # filePath = "./ieekeypoints.py"
    if not isfile(imgPath + ".png"):
        #ls = subprocess.run(['ls', '-a'], capture_output=True, text=True).stdout.strip("\n")
        ls = subprocess.run(
            [str(blenderPath), "--background", "--verbose", str(0), "--python", str(filePath), "--", "--path",
             str(path), "--model_folder",
             str(model_folder), "--label_folder", str(label_folder), "--pose_folder", str(pose_folder), "--pose_file",
             str(pose_file[m]), "--label_file", str(label_file[m]), "--model_file", str(model_file[m]), "--imgPath",
             # str(imgPath)], stdout=subprocess.PIPE)
             str(imgPath)], capture_output=True, text=True).stderr.strip("\n")
        #print(ls)
        #print(process.stdout)
        #print(process.stderr)
    # process.wait()
    #print("Image Generation", str(t2-t1)[0:5])
    #print(imgPath)
    img = cv2.imread(imgPath + ".png")
    if img is None:
        print("image not found, not processed")
        return imgPath, False
    if len(img) > 128:
        faceFound = processImage(imgPath + ".png", join(outPath, "IEEPackage/clsdata/mmod_human_face_detector.dat"))
    else:
        faceFound = True
    # generator = ieeKP.IEEKPgenerator(model_folder, pose_folder, label_folder)
    # imgPath = generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
    #                                                   cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
    #                                                   label_file[m])
    #print("Image processing", str(time.time()-t2)[0:5])
    return imgPath, faceFound


def processImage(imgPath, dlibPath):
    img = cv2.imread(imgPath)
    npPath = join(dirname(imgPath), basename(imgPath).split(".png")[0] + ".npy")
    configFile = np.load(npPath, allow_pickle=True)
    labelFile = configFile.item()['label']

    face_detector = dlib.cnn_face_detection_model_v1(dlibPath)
    faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    # img = new_img
    # labelFile = np.array(label_arr)
    width = img.shape[1]
    height = img.shape[0]
    mouth = [6, 7, 8, 23, 24, 25, 26]
    mouth = [32, 34, 36, 49, 52, 55, 58]

    for KP in mouth:
        x_p = labelFile[KP][0]
        y_p = img.shape[0] - labelFile[KP][1]
        px = x_p
        py = y_p
        if KP == 32:
            p1x = px
            p1y = py
        elif KP == 36:
            p2x = px
            p2y = py
        elif KP == 34:
            p7x = px
            p7y = py
        elif KP == 58:
            p3x = px
            p3y = py
        elif KP == 52:
            p4x = px
            p4y = py
        elif KP == 49:
            p5x = px
            p5y = py
        elif KP == 55:
            p6x = px
            p6y = py
    # new_img = putMask(imgPath, img, [p1x, p2x, p3x, p4x, p5x, p6x, p7x], [p1y, p2y, p3y, p4y, p5y, p6y, p7y])
    # cv2.imwrite(imgPath, new_img)
    # img = new_img
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return
    if len(faces) < 1:
        return False
    big_face = -np.inf
    mx, my, mw, mh = 0, 0, 0, 0
    for face in faces:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        if w * h > big_face:
            big_face = w * h
            mx, my, mw, mh = x, y, w, h
    sw_0 = max(mx - 25 // 2, 0)
    sw_1 = min(mx + mw + 25 // 2, new_img.shape[1])  # empirical
    sh_0 = max(my - 25 // 2, 0)
    sh_1 = min(my + mh + 25 // 2, new_img.shape[0])  # empirical
    assert sh_1 > sh_0
    assert sw_1 > sw_0
    label_arr = []
    iee_labels = [18, 22, 23, 27, 28, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 69, 43, 44, 45, 46, 47, 48, 70, 49, 52,
                  55, 58]
    for ky in iee_labels:
        if ky in labelFile:
            coord = [labelFile[ky][0], new_img.shape[0] - labelFile[ky][1]]
            label_arr.append(coord)  # (-1,-1) means the keypoint is invisible
        else:
            label_arr.append([0, 0])  # label does not exist
    new_label = np.zeros_like(np.array(label_arr))
    new_label[:, 0] = np.array(label_arr)[:, 0] - sw_0
    new_label[:, 1] = np.array(label_arr)[:, 1] - sh_0
    new_label[new_label < 0] = 0
    new_label[np.array(label_arr)[:, 0] == -1, 0] = -1
    new_label[np.array(label_arr)[:, 1] == -1, 1] = -1
    big_face = new_img[sh_0:sh_1, sw_0:sw_1]
    width_resc = float(128) / big_face.shape[0]
    height_resc = float(128) / big_face.shape[1]
    new_label2 = np.zeros_like(new_label)
    new_label2[:, 0] = new_label[:, 0] * width_resc
    new_label2[:, 1] = new_label[:, 1] * height_resc
    labelFile = new_label2
    new_img = cv2.resize(big_face, (128, 128), interpolation=cv2.INTER_CUBIC)
    x_data = new_img
    x_data = np.repeat(x_data[:, :, np.newaxis], 3, axis=2)
    img = x_data
    cv2.imwrite(imgPath, img)
    return True



def labelImage(imgPath):
    margin1 = 10.0
    margin2 = -10.0
    margin3 = 10.0
    margin4 = -10.0
    configPath = join(dirname(imgPath), basename(imgPath).split(".png")[0] + ".npy")
    configFile = np.load(configPath, allow_pickle=True)
    configFile = configFile.item()
    HP1 = configFile['config']['head_pose'][0]
    HP2 = configFile['config']['head_pose'][1]
    originalDst = None
    if HP1 > margin1:
        if HP2 > margin3:
            originalDst = "BottomRight"
        elif HP2 < margin4:
            originalDst = "BottomLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "BottomCenter"
    elif HP1 < margin2:
        if HP2 > margin3:
            originalDst = "TopRight"
        elif HP2 < margin4:
            originalDst = "TopLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "TopCenter"
    elif margin2 <= HP1 <= margin1:
        if HP2 > margin3:
            originalDst = "MiddleRight"
        elif HP2 < margin4:
            originalDst = "MiddleLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "MiddleCenter"
    if originalDst is None:
        print("cannot label img:", imgPath)
    return originalDst


def doImage(imgPath, caseFile, centroidHM):

    layersHM, entropy = HeatmapModule.generateHeatMap(imgPath, caseFile["DNN"], caseFile["datasetName"], caseFile["outputPath"],
                                        False, None, None, caseFile["imgExt"], None)
    lable = labelImage(imgPath)
    DNNResult, pred = testModule.testModelForImg(caseFile["DNN"], lable, imgPath, caseFile)
    #if imgPath in Dist_Dict:
    if centroidHM is None:
        dist = 0
    else:
        dist = HeatmapModule.doDistance(centroidHM, layersHM[int(caseFile["selectedLayer"].replace("Layer", ""))], "Euc")
    #Dist_Dict[imgPath] = dist
    return entropy, DNNResult, pred, lable, dist, layersHM


def setNewX(x, paramDict, paramNameList, param, val1, val2, nVariables=nVar):
    for j in range(0, nVariables):
        minVal = min(paramDict[paramNameList[j]])
        maxVal = max(paramDict[paramNameList[j]])
        #if j == 19:
        #    maxVal+=0.99 #we round down the facemodel value
        x[j] = random.uniform(minVal, maxVal)
        for z in range(0, len(param)):
            if j == (int(param[z]) - 1):
                #if j == 19:
                #    val2[z] = val2[z] + 0.99
                if float(val1[z]) < minVal:
                    val1[z] = minVal
                if float(val2[z]) > maxVal:
                    val2[z] = maxVal
                #print(val1[z], val2[z])
                if float(val1[z]) > float(val2[z]):
                    "error in parameters settings"
                x[j] = random.uniform(float(val1[z]), float(val2[z]))

        #print(paramNameList[j], x[j])
    return x

def getParamVals():
    param_list = ["cam_look_0", "cam_look_1", "cam_look_2", "cam_loc_0", "cam_loc_1", "cam_loc_2",
                  "lamp_loc_0", "lamp_loc_1", "lamp_loc_2", "lamp_direct_0", "lamp_direct_1", "lamp_direct_2",
                  "lamp_color_0", "lamp_color_1", "lamp_color_2", "head_0", "head_1", "head_2", "lamp_energy"]
    # TrainingSet parameters (min - max)
    cam_dirL = [-0.10, -4.67, -1.69]
    # cam_dirL = [-0.08, -4.29, -1.27] # constant
    cam_dirU = [-0.08, -4.29, -1.27]
    cam_locL = [0.261, -5.351, 14.445]
    # cam_locL = [0.293, -5.00, 14.869] # constant
    cam_locU = [0.293, -5.00, 14.869]  # constant
    lamp_locL = [0.361, -5.729, 16.54]
    # lamp_locL = [0.381, -5.619, 16.64] # constant
    lamp_locU = [0.381, -5.619, 16.64]  # constant
    lamp_colL = [1.0, 1.0, 1.0]  # constant
    lamp_colU = [1.0, 1.0, 1.0]  # constant
    lamp_dirL = [0.873, -0.87, 0.698]  # constant
    lamp_dirU = [0.873, -0.87, 0.698]  # constant
    lamp_engL = 1.0  # constant
    lamp_engU = 1.0  # constant
    headL = [-41.86, -79.86, -64.30]
    headU = [36.87, 75.13, 51.77]
    faceL = 0
    faceU = 8
    # TestSet parameters (min - max)
    # headL = [-32.94, -88.10, -28.53]
    # headU = [33.50, 74.17, 46.17]
    # fixing HP_2
    # headL = [-32.94, -88.10, -0.000001]
    # headU = [33.50, 74.17, 0]
    return param_list, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, lamp_colL, lamp_colU, lamp_dirL, \
           lamp_dirU, lamp_engL, lamp_engU, headL, headU, faceL, faceU


def getParamVals_2():  # IEE_V2
    param_list = ["head0", "head1", "head2", "lampcol0", "lampcol1",
                  "lampcol2", "lamploc0", "lamploc1", "lamploc2", "lampdir0", "lampdir1", "lampdir2", "cam",
                  "age", "hue", "iris", "sat", "val", "freckle", "oil", "veins", "eyecol", "gender"]

    headL = [-50, -50, -50]
    headU = [50, 50, 50]
    lamp_colL = [0, 0, 0]
    lamp_colU = [1, 1, 1]
    lamp_locL = [0.361, -5.729, 16.54]
    lamp_locU = [0.381, -5.619, 16.64]
    lamp_dirL = [0.843, -0.85, 0.598]
    lamp_dirU = [0.893, -0.89, 0.798]
    camL = -0.5
    camU = 0.5
    ageL = -1
    hueL = irisL = satL = valL = freckleL = oilL = veinsL = eye_colL = genderL = 0
    hueU = irisU = satU = valU = freckleU = oilU = veinsU = ageU = 1
    genderU = 1
    eye_colU = 3  # math.floor(eye_colU)
    return param_list, headL, headU, lamp_colL, lamp_colU, lamp_locL, lamp_locU, lamp_dirL, lamp_dirU, camL, camU, \
           ageL, ageU, hueL, hueU, irisL, irisU, satL, satU, valL, valU, freckleL, freckleU, oilL, oilU, veinsL, \
           veinsU, eye_colL, eye_colU, genderL, genderU


def setX(size, ID):
    _, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, lamp_colL, lamp_colU, lamp_dirL, \
    lamp_dirU, lamp_engL, lamp_engU, headL, headU, faceL, faceU = getParamVals()
    xl = []
    if ID == "L":
        for i in range(0, size):
            for c in cam_dirL:
                xl.append(c)
            for c in cam_locL:
                xl.append(c)
            for c in lamp_locL:
                xl.append(c)
            for c in lamp_colL:
                xl.append(c)
            for c in lamp_dirL:
                xl.append(c)
            xl.append(lamp_engL)
            for c in headL:
                xl.append(c)
            xl.append(faceL)
    elif ID == "U":
        xl = []
        for i in range(0, size):
            for c in cam_dirU:
                xl.append(c)
            for c in cam_locU:
                xl.append(c)
            for c in lamp_locU:
                xl.append(c)
            for c in lamp_colU:
                xl.append(c)
            for c in lamp_dirU:
                xl.append(c)
            xl.append(lamp_engU)
            for c in headU:
                xl.append(c)
            xl.append(faceU)
    elif ID == "R":
        xl = []
        for i in range(0, size):
            for z in range(0, len(cam_dirU)):
                xl.append(random.uniform(cam_dirL[z], cam_dirU[z]))
            for z in range(0, len(cam_locU)):
                xl.append(random.uniform(cam_locL[z], cam_locU[z]))
            for z in range(0, len(lamp_locU)):
                xl.append(random.uniform(lamp_locL[z], lamp_locU[z]))
            for z in range(0, len(lamp_colU)):
                xl.append(random.uniform(lamp_colL[z], lamp_colU[z]))
            for z in range(0, len(lamp_dirU)):
                xl.append(random.uniform(lamp_dirL[z], lamp_dirU[z]))
            xl.append(random.uniform(lamp_engL, lamp_engU))
            for z in range(0, len(headU)):
                xl.append(random.uniform(headL[z], headU[z]))
            xl.append(random.uniform(faceL, faceU))
    return np.array(xl)


def setX_2(size, ID):
    _, headL, headU, lamp_colL, lamp_colU, lamp_locL, lamp_locU, lamp_dirL, lamp_dirU, camL, camU, \
    ageL, ageU, hueL, hueU, irisL, irisU, satL, satU, valL, valU, freckleL, freckleU, oilL, oilU, veinsL, \
    veinsU, eye_colL, eye_colU, genderL, genderU = getParamVals_2()
    xl = []
    if ID == "L":
        for i in range(0, size):
            for c in headL:
                xl.append(c)
            for c in lamp_colL:
                xl.append(c)
            for c in lamp_locL:
                xl.append(c)
            for c in lamp_dirL:
                xl.append(c)
            xl.append(camL)
            xl.append(ageL)
            xl.append(hueL)
            xl.append(irisL)
            xl.append(satL)
            xl.append(valL)
            xl.append(freckleL)
            xl.append(oilL)
            xl.append(veinsL)
            xl.append(eye_colL)
            xl.append(genderL)
    elif ID == "U":
        xl = []
        for i in range(0, size):
            for c in headU:
                xl.append(c)
            for c in lamp_colU:
                xl.append(c)
            for c in lamp_locU:
                xl.append(c)
            for c in lamp_dirU:
                xl.append(c)
            xl.append(camU)
            xl.append(ageU)
            xl.append(hueU)
            xl.append(irisU)
            xl.append(satU)
            xl.append(valU)
            xl.append(freckleU)
            xl.append(oilU)
            xl.append(veinsU)
            xl.append(eye_colU)
            xl.append(genderU)
    elif ID == "R":
        xl = []
        for i in range(0, size):
            for z in range(0, len(headU)):
                xl.append(random.uniform(headL[z], headU[z]))
            for z in range(0, len(lamp_colU)):
                xl.append(random.uniform(lamp_colL[z], lamp_colU[z]))
            for z in range(0, len(lamp_locU)):
                xl.append(random.uniform(lamp_locL[z], lamp_locU[z]))
            for z in range(0, len(lamp_dirU)):
                xl.append(random.uniform(lamp_dirL[z], lamp_dirU[z]))
            xl.append(random.uniform(camL, camU))
            xl.append(random.uniform(ageL, ageU))
            xl.append(random.uniform(hueL, hueU))
            xl.append(random.uniform(irisL, irisU))
            xl.append(random.uniform(satL, satU))
            xl.append(random.uniform(valL, valU))
            xl.append(random.uniform(freckleL, freckleU))
            xl.append(random.uniform(oilL, oilU))
            xl.append(random.uniform(veinsL, veinsU))
            xl.append(random.uniform(eye_colL, eye_colU))
            xl.append(random.uniform(genderL, genderU))
    return np.array(xl)


class DataSupplier(object):
    #if using_gm is true, convert a coordinate to gaussian map
    def __init__(self, file_path, batch_size, using_gm, pin_memory, max_num=0):
        self.dataformer = DataTransformer()
        self.using_gm = using_gm
        self.pin_memory = pin_memory
        self.file_path = file_path
        self.batch_size = batch_size
        self.max_num = max_num
        return

    def get_data_iters(self):
        x_data, y_data = get_data(self.file_path, self.max_num)
        transform_func = self.dataformer.tranform_basic()
        data_set = KPDataset(x_data, y_data, using_gm=self.using_gm, transforms=transform_func)
        data_iter = DataIter(data_set, 0, self.batch_size, self.pin_memory) #we split training/testing set before diving into DataSupplier
        iters, __ = data_iter.get_iters()
        return iters, len(x_data)
        #return iters

#class DataSupplier(object):
#    def __init__(self, using_gm=True):
#        self.dataformer = DataTransformer()
#        self.using_gm = using_gm
#        return

#    def get_train_iter(self, dataPath):
#        X_train, y_train, imageList = get_data(dataPath)
#        trans = self.dataformer.tranform_basic()
#        train_set = KPDataset(X_train, y_train, using_gm=self.using_gm, transforms=trans)
#        di = DataIter(train_set, validRatio, pinMemory)
        # return di.get_iters(True)[0], di.get_iters(True)[1], imageList
#        return di.get_iters(True)

#    def get_test_iter(self, dataPath):  # dont call me when using ieedata
#        X_test, y_test, imageList = get_data(dataPath)
#        trans = self.dataformer.tranform_basic()
#        test_set = KPDataset(X_test, y_test, using_gm=False, transforms=trans)
#        di = DataIter(test_set, 0, False)
#        print(di.get_iters(False))
#        return di.get_iters(False)[0], di.get_iters(False)[1], imageList
def getFileList(dirPath):
    imgList = []
    for src_dir, dirs, files in os.walk(dirPath):
        for file_ in files:
            if file_.endswith(".pt"):
                imgList.append(file_)
    return imgList
class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(PathImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def cleanMake(path, flag):
    if not exists(path):
        makedirs(path)
    else:
        if flag:
            rmtree(path)
            makedirs(path)

def get_all_pngs(folder):
    folder_f = folder + "/*.png"
    f_list = glob.glob(folder_f)
    return f_list


def get_label(file_name):
    print("label file_name: ", file_name)
    data = np.load(file_name, allow_pickle=True)
    data = data.item()
    label_value = np.zeros((1, cood_num * 2))

    idx = 0
    for ky in labels:
        coods = np.array(data[ky])
        # print("ky: ", ky, "coods: ", coods.shape)
        coods = coods[coods[:, 0].argsort()]
        for co in coods:
            label_value[0, idx] = co[1]
            label_value[0, idx + 1] = iee_img_height - co[2]
            idx += 2

    return label_value


def crop_img_lab(idx, faces, img, kps, evidence_path):
    new_label = np.zeros_like(kps)

    big_face = -np.inf
    mx, my, mw, mh = 0, 0, 0, 0
    for face in faces:  # we only need to consider one face, fix this later
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        if w * h > big_face:
            big_face = w * h
            mx, my, mw, mh = x, y, w, h

    sx_0 = max(mx - w_tol // 2, 0)
    sx_1 = min(sx_0 + mw + w_tol, iee_img_width)

    sy_0 = max(my - h_tol // 2, 0)
    sy_1 = min(sy_0 + mh + h_tol * 2, iee_img_height)

    assert sy_1 > sy_0
    assert sx_1 > sx_0

    new_img = img[sy_0:sy_1, sx_0:sx_1]
    tmp_h, tmp_w = new_img.shape
    new_label[0, ::2] = kps[0, ::2] - sx_0
    new_label[0, 1::2] = kps[0, 1::2] - sy_0
    new_label[new_label < 0] = 0

    new_img = cv2.resize(new_img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    width_resc = float(target_size) / tmp_w
    height_resc = float(target_size) / tmp_h

    new_label[0, ::2] = new_label[0, ::2] * (width_resc)
    new_label[0, 1::2] = new_label[0, 1::2] * (height_resc)

    # new_label[0, [-4, -3, -2, -1]] = sx_0, sy_0, width_resc, height_resc
    # new_img = new_img.reshape(1,int(target_size*target_size))

    return new_img, new_label


def get_img(apng):
    img = cv2.imread(apng)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_face_detector(weight_path):
    face_detector = dlib.cnn_face_detection_model_v1(weight_path)
    return face_detector


def get_data_kaggle(f_path, btest=False):
    df = pd.read_csv(f_path)
    df = df.fillna(-1)
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' '))
    X = np.vstack(df['Image'].values)
    X = X.astype(np.float32)
    X = X / 255.  # scale pixel values to [0, 1]
    X = X.reshape(-1, 1, 96, 96)  # return each images as 1 x 96 x 96

    if not btest:
        y = df[df.columns[:-1]].values
        y = y.astype(np.float32)
    else:
        y = np.zeros((len(X)))

    return X, y


def get_data(f_path, btest=False):
    data = np.load(f_path, allow_pickle=True)
    data = data.item()

    X = data["Image"]
    X = X.astype(np.float32)
    X = X / 255.
    X = X.reshape(-1, 1, width, height)

    y = data["Label"]
    y = y.astype(np.float32)

    imageList = data["Origin"]
    return X, y, imageList


class KPDataset(Dataset):
    def __init__(self, X, y, using_gm=True, transforms=None):
        self.X = X
        self.y = y
        self.using_gm = using_gm
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.X[index]
        kps = self.y[index]
        kps = kps.reshape(1, len(kps))

        if self.transforms:
            img = self.transforms(img)

        if self.using_gm:
            gm = GaussianMap(kps, width, height, sigma)
            gm_kps = gm.create_heatmaps() * gaussian_scale
            kps = {"kp": kps, "gm": gm_kps}
        return img, kps

    def __len__(self):
        return len(self.X)


class ToTensor(object):
    def __call__(self, img):
        # imagem numpy: C x H x W
        # imagem torch: C X H X W          
        img = img.transpose((0, 1, 2))
        return torch.from_numpy(img)


class CloneArray(object):
    def __call__(self, img):
        img = img.repeat(3, axis=0)
        return img


class DataTransformer(object):
    def __init__(self):
        # self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        # self.imagenet_std  = np.array([0.229, 0.224, 0.225])
        return

    def tranform_basic(self):
        # trans =  transforms.Compose([CloneArray(),
        #                             ToTensor(),
        #                             transforms.Normalize(self.imagenet_mean, self.imagenet_std)])
        trans = transforms.Compose([ToTensor()])
        return trans


class DataIter(object):
    def __init__(self, dataset, valid_ratio, pin_memory=False):
        self.pin_memory = pin_memory
        self.dataset = dataset
        self.num = len(dataset)
        self.valid_ratio = valid_ratio
        return

    def get_samplers(self, shuffle):
        indices = list(range(self.num))
        if shuffle:
            np.random.seed(data_random_seed)
            np.random.shuffle(indices)

        if self.valid_ratio > 0:
            split = int(np.floor(self.valid_ratio * self.num))
            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            return train_sampler, valid_sampler
        else:
            train_sampler = SubsetRandomSampler(indices)
            return train_sampler, None

    def get_iters(self, shuffle=True):
        if self.valid_ratio > 0:
            train_sampler, valid_sampler = self.get_samplers(shuffle)
            train_iter = DataLoader(self.dataset, batch_size=batch_size,
                                    sampler=train_sampler, pin_memory=self.pin_memory)
            valid_iter = DataLoader(self.dataset, batch_size=batch_size,
                                    sampler=valid_sampler, pin_memory=self.pin_memory)
            return train_iter, valid_iter
        else:
            train_iter = DataLoader(self.dataset, batch_size=batch_size,
                                    pin_memory=self.pin_memory, shuffle=False)
            return train_iter, None


def loadTestData(testdataPath: str, batchSize, workersCount, datasetName):
    testData = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=testdataPath, transform=setupTransformer(datasetName)),
        batch_size=batchSize, shuffle=True,
        num_workers=workersCount)
    return testData


def loadTrainData(bagPath, caseFile_, imagesList):
    global caseFile
    caseFile = caseFile_
    trainDataSet = caseFile["trainDataSet"]
    datasetName = caseFile["datasetName"]
    if not exists(bagPath):
        os.makedirs(bagPath)
    imgClasses = trainDataSet.dataset.classes
    for imgclass in imgClasses:
        if not exists(join(bagPath, imgclass)):
            os.makedirs(join(bagPath, imgclass))
    imgLst = collectData(bagPath, imagesList)
    #if retrainMode == "BL1":
    #    imgLst = BL1_Data(improvSet, bagPath, U, clsPath, net, datasetName)
    #elif retrainMode == "BL4":
    #    imgLst = BL4_Data(improvSet, bagPath, net, datasetName)
    ts = datasets.ImageFolder(root=bagPath, transform=setupTransformer(datasetName))
    return ts, imgLst, caseFile


def IEE_HUDD(caseFile, outputSet):
    components = caseFile["components"]
    trainDataNpy = caseFile["trainDataNpy"]
    improveDataNpy = caseFile["improveDataNpy"]
    unsafeDataSet_X = []
    retrainDataSet_X = []
    unsafeDataSet_Y = []
    retrainDataSet_Y = []
    trainDataset = np.load(trainDataNpy, allow_pickle=True)
    trainDataset = trainDataset.item()
    a_data = trainDataset["data"]
    b_data = trainDataset["label"]
    for i in range(0, len(a_data)):
        retrainDataSet_X.append(a_data[i])
        retrainDataSet_Y.append(b_data[i])
    ISdataset = np.load(improveDataNpy, allow_pickle=True)
    ISdataset = ISdataset.item()
    x_data = ISdataset["data"]
    y_data = ISdataset["label"]
    selected = list()
    totalImages = []
    numClusters = 0
    if "retrainSet" not in caseFile:
        for component in components:
            newPath = join(caseFile["outputPathOriginal"], component, caseFile["RCC"],
                                   "ClusterAnalysis_" + caseFile["clustMode"], "Assignments",
                                   caseFile["assignMode"], caseFile["selectionMode"])
            clsWithAssImages = torch.load(join(newPath, "clusterwithAssignedImages.pt"))
            caseFile["assignPTFile"] = join(newPath, "clusterwithAssignedImages.pt")
            clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3) #U4/U5
            # clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.1) #U7/U8
            for clusterID in clsWithAssImages['clusters']:
                closestClusterName = torch.load(join(newPath, "improveRCCDists", "closestClusterName.pt"))
                closestClusterDist = torch.load(join(newPath, "improveRCCDists", "closestClusterDist.pt"))
                if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                    unsafeImages = []
                    clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                    if clustLen > 1:
                        toNormalize = list()
                        imagesList = list()
                        breakFlag = False
                        for order in range(0, len(clsWithAssImages['clusters'])):
                            for _ in caseFile["retrainList"]:
                                candidateImage = min(closestClusterDist[order].keys(),
                                                     key=(lambda k: closestClusterDist[order][k]))
                                candidateClusterID = closestClusterName[order][candidateImage]
                                dif = closestClusterDist[order][candidateImage]
                                closestClusterDist[order][candidateImage] = 1e9
                                fileFullName = basename(candidateImage)
                                fileIndxName = fileFullName.split(".")[0]
                                fileIndex = int(fileIndxName.split("I")[1]) - 1
                                if candidateClusterID == clusterID:
                                    if len(unsafeImages) < clusterUCs[clusterID]:
                                        unsafeDataSet_X.append(x_data[fileIndex])
                                        unsafeDataSet_Y.append(y_data[fileIndex])
                                        retrainDataSet_X.append(x_data[fileIndex])
                                        retrainDataSet_Y.append(y_data[fileIndex])
                                        imagesList.append(fileIndex)
                                        selected.append(fileIndex)
                                        unsafeImages.append(fileIndex)
                                        totalImages.append(fileIndex)
                                        toNormalize.append(dif)
                                    else:
                                        breakFlag = True
                                if breakFlag:
                                    break
                        if len(imagesList) > 1:
                            probList = list()
                            probList2 = list()
                            probList3 = list()
                            for i, val in enumerate(toNormalize):
                                probList.append(1 - (val / sum(toNormalize)))
                            for i, val in enumerate(probList):
                                if i == 0:
                                    offset = 0
                                else:
                                    offset = probList2[i - 1]
                                probList2.append(val + offset)
                            for i, val in enumerate(probList2):
                                if (max(probList2) - min(probList2)) == 0:
                                    print(unsafeImages, toNormalize, probList, probList2)
                                probList3.append((val - min(probList2)) / (max(probList2) - min(probList2)))
                            while len(unsafeImages) < Ub:
                                randNum = random.uniform(0, 1)
                                for z in range(0, len(probList)):
                                    if randNum < probList[z]:
                                        if len(unsafeImages) < Ub:
                                            unsafeDataSet_X.append(x_data[imagesList[z]])
                                            unsafeDataSet_Y.append(y_data[imagesList[z]])
                                            retrainDataSet_X.append(x_data[imagesList[z]])
                                            retrainDataSet_Y.append(y_data[imagesList[z]])
                                            unsafeImages.append(imagesList[z])
                                            totalImages.append(imagesList[z])
                        else:
                            while len(unsafeImages) < Ub:
                                fileFullName = basename(clsWithAssImages['clusters'][clusterID]['assigned'][0])
                                fileIndxName = fileFullName.split(".")[0]
                                fileIndex = int(fileIndxName.split("I")[1]) - 1
                                unsafeDataSet_X.append(x_data[fileIndex])
                                unsafeDataSet_Y.append(y_data[fileIndex])
                                retrainDataSet_X.append(x_data[fileIndex])
                                retrainDataSet_Y.append(y_data[fileIndex])
                                unsafeImages.append(fileIndex)
                                totalImages.append(fileIndex)
            clusterDistrib = list()
            for clusterID in clsWithAssImages['clusters']:
                if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                    numClusters += 1
                    clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                    clusterDistrib.append(clustLen)
            #print(component, "Assigned Images to Clusters Distribution:", clusterDistrib)
        retrainDataSet_X1 = np.array(retrainDataSet_X)
        retrainDataSet_Y1 = np.array(retrainDataSet_Y)
        print("Total Number of Clusters with Assigned images:", numClusters)
        print("Size of UnsafeSet:", len(selected))
        print("Size of Bagged UnsafeSet:", len(totalImages))
        print("Size of RetrainSet:", str(len(retrainDataSet_X)))
        retrainDataSet = {"data": retrainDataSet_X1, "label": retrainDataSet_Y1}
        np.save(outputSet, retrainDataSet)



def IEE_BL2(caseFile, outputSet):
    components = caseFile["components"]
    trainDataNpy = caseFile["trainDataNpy"]
    improveDataNpy = caseFile["improveDataNpy"]
    retrainDataSet_X = []
    retrainDataSet_Y = []
    trainDataset = np.load(trainDataNpy, allow_pickle=True)
    trainDataset = trainDataset.item()
    a_data = trainDataset["data"]
    b_data = trainDataset["label"]
    for i in range(0, len(a_data)):
        retrainDataSet_X.append(a_data[i])
        retrainDataSet_Y.append(b_data[i])
    ISdataset = np.load(improveDataNpy, allow_pickle=True)
    ISdataset = ISdataset.item()
    x_data = ISdataset["data"]
    y_data = ISdataset["label"]
    BLDataSet_X = []
    BLDataSet_Y = []
    for i in range(0, len(a_data)):
        BLDataSet_X.append(a_data[i])
        BLDataSet_Y.append(b_data[i])
    numClusters = 0
    totalSelectedImages = []
    selected = []
    totalUcCombined = 0
    totalUbCombined = 0
    for component in components:
        newPath = join(caseFile["outputPathOriginal"], component, caseFile["RCC"],
                               "ClusterAnalysis_" + caseFile["clustMode"], "Assignments",
                               caseFile["assignMode"], caseFile["selectionMode"])
        clsWithAssImages = torch.load(join(newPath, "clusterwithAssignedImages.pt"))
        caseFile["assignPTFile"] = join(newPath, "clusterwithAssignedImages.pt")
        clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3)
        totalUcCombined += totalUc
        totalUbCombined += totalUb
        clusterDistrib = list()
        for clusterID in clsWithAssImages['clusters']:
            if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                clusterDistrib.append(clustLen)
                numClusters += 1
    if "retrainSet" not in caseFile:
        totalSelectedComponentImages = []
        for i in range(0, totalUcCombined):
            imgID = random.randint(0, len(x_data) - 1)
            BLDataSet_X.append(x_data[imgID])
            BLDataSet_Y.append(y_data[imgID])
            selected.append(imgID)
            totalSelectedImages.append(imgID)
            totalSelectedComponentImages.append(imgID)
        while len(totalSelectedComponentImages) < totalUbCombined:
            imgID = random.randint(0, len(totalSelectedComponentImages) - 1)
            BLDataSet_X.append(x_data[totalSelectedComponentImages[imgID]])
            BLDataSet_Y.append(y_data[totalSelectedComponentImages[imgID]])
            totalSelectedImages.append(totalSelectedComponentImages[imgID])
            totalSelectedComponentImages.append(totalSelectedComponentImages[imgID])
        print("Total Number of Clusters with Assigned images:", numClusters)
        print("Total UnsafeSet:", len(selected))
        print("Total Bagged UnsafeSet:", len(totalSelectedImages))
        print("Size of RetrainSet:", str(len(BLDataSet_X)))
        BLDataSet_X1 = np.array(BLDataSet_X)
        BLDataSet_Y1 = np.array(BLDataSet_Y)
        BLDataSet = {"data": BLDataSet_X1, "label": BLDataSet_Y1}
        np.save(outputSet, BLDataSet)


def IEE_BL1(caseFile, outputSet, errList):
    components = caseFile["components"]
    trainDataNpy = caseFile["trainDataNpy"]
    improveDataNpy = caseFile["improveDataNpy"]
    retrainDataSet_X = []
    retrainDataSet_Y = []
    trainDataset = np.load(trainDataNpy, allow_pickle=True)
    trainDataset = trainDataset.item()
    a_data = trainDataset["data"]
    b_data = trainDataset["label"]
    for i in range(0, len(a_data)):
        retrainDataSet_X.append(a_data[i])
        retrainDataSet_Y.append(b_data[i])
    ISdataset = np.load(improveDataNpy, allow_pickle=True)
    ISdataset = ISdataset.item()
    x_data = ISdataset["data"]
    y_data = ISdataset["label"]
    BLDataSet_X = []
    BLDataSet_Y = []
    for i in range(0, len(a_data)):
        BLDataSet_X.append(a_data[i])
        BLDataSet_Y.append(b_data[i])
    numClusters = 0
    totalSelectedImages = []
    imageListx = pd.read_csv(caseFile["improveCSV"])
    imageList = []
    selected = []
    for index, row in imageListx.iterrows():
        result = "Correct"
        if row["result"] == "Wrong":
    #        #if row["worst_component"] == component:
                result = "Wrong"
        imageList.append(result)
    totalUcCombined = 0
    totalUbCombined = 0
    for component in components:
        newPath = join(caseFile["outputPathOriginal"], component, caseFile["RCC"],
                               "ClusterAnalysis_" + caseFile["clustMode"], "Assignments",
                               caseFile["assignMode"], caseFile["selectionMode"])
        clsWithAssImages = torch.load(join(newPath, "clusterwithAssignedImages.pt"))
        caseFile["assignPTFile"] = join(newPath, "clusterwithAssignedImages.pt")
        clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3)
        totalUcCombined += totalUc
        totalUbCombined += totalUb
        clusterDistrib = list()
        for clusterID in clsWithAssImages['clusters']:
            if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                clusterDistrib.append(clustLen)
                numClusters += 1
    if "retrainSet" not in caseFile:
        totalSelectedComponentImages = []
        totalSelectedComponentImages2 = []
        for i in range(0, totalUcCombined):
            imgID = random.randint(0, len(imageList) - 1)
            totalSelectedComponentImages.append(imgID)
        print("Total Selected:", len(totalSelectedComponentImages))
        for imgID in totalSelectedComponentImages:
            if errList[imgID] == "Wrong":
                BLDataSet_X.append(x_data[imgID])
                BLDataSet_Y.append(y_data[imgID])
                totalSelectedImages.append(imgID)
                selected.append(imgID)
                totalSelectedComponentImages2.append(imgID)
        print("Total Failing:", len(totalSelectedComponentImages2))
        if len(totalSelectedComponentImages2) > 0:
            while len(totalSelectedComponentImages2) < totalUbCombined:
                imgID = random.randint(0, len(totalSelectedComponentImages2) - 1)
                BLDataSet_X.append(x_data[imgID])
                BLDataSet_Y.append(y_data[imgID])
                totalSelectedImages.append(imgID)
                totalSelectedComponentImages2.append(imgID)
        print("Total Number of Clusters with Assigned images:", numClusters)
        print("Total UnsafeSet:", len(totalSelectedImages))
        print("Total Bagged UnsafeSet:", len(totalSelectedImages))
        print("Size of RetrainSet:", str(len(BLDataSet_X)))
        BLDataSet_X1 = np.array(BLDataSet_X)
        BLDataSet_Y1 = np.array(BLDataSet_Y)
        BLDataSet = {"data": BLDataSet_X1, "label": BLDataSet_Y1}
        np.save(outputSet, BLDataSet)


def loadIEETrainData(caseFile, outputSet, errList):
    if caseFile["retrainMode"].startswith("HUDD"):
        IEE_HUDD(caseFile, outputSet)
    elif caseFile["retrainMode"] == "BL2":
        IEE_BL2(caseFile, outputSet)
    elif caseFile["retrainMode"] == "BL1":
        IEE_BL1(caseFile, outputSet, errList)
    else:
        IEE_fineTune(caseFile, outputSet)


def generateSSEData():
    global clsWithAssImages
    print("Closest-SSE + Random Bagging")
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3) #U4/U5
    # clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.1) #U7/U8
    outputPath = caseFile["outputPath"]
    selectedLayer = caseFile["selectedLayer"]
    area = basename(dirname(outputPath))
    npyPath = join(dirname(outputPath), "ieeimprove.npy")
    trainHeatmaps = join(caseFile["filesPath"], "trainHeatmaps", selectedLayer)
    testHM, _ = HeatmapModule.collectHeatmaps(caseFile["filesPath"], selectedLayer)
    totalImages = []
    for clusterID in clsWithAssImages['clusters']:
        if 'assigned' in clsWithAssImages['clusters'][clusterID]:
            unsafeImages = []
            selectedSSE = {}
            clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
            if clustLen > 0:
                Uc = clusterUCs[clusterID]
                if Uc < clustLen:
                    for img in clsWithAssImages['clusters'][clusterID]['assigned']:
                        SSE = 0
                        imgExt = "." + str(basename(img).split(".")[1])
                        imgName = str(basename(img).split(".")[0])
                        imgClass = str(basename(dirname(img)))
                        HMFile = join(trainHeatmaps, imgName + "_" + imgClass + ".pt")
                        heatMap = HeatmapModule.safeHM(HMFile, int(selectedLayer.replace("Layer", "")), img, net,
                                                       caseFile["datasetName"], outputPath, False, area, npyPath, imgExt, None)
                        for testImage in clsWithAssImages['clusters'][clusterID]['members']:
                           SSE += HeatmapModule.doDistance(heatMap, testHM[testImage], caseFile["metric"]) ** 2
                        selectedSSE[img] = SSE
                        bestSSEimage = min(selectedSSE.keys(), key=(lambda k: selectedSSE[k]))
                        unsafeImages.append(bestSSEimage)
                        totalImages.append(bestSSEimage)
                        del selectedSSE[bestSSEimage]
    return totalImages


def generateSmartBaggingEntropy():
    global clsWithAssImages
    #print("SmartBagging")
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3) #U4/U5
    #clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.1) #U7/U8
    totalImages = []
    unsafeImages = []
    closestClusterDist = torch.load(join(caseFile["improveRCCDists"], "closestClusterDist.pt"))
    toNormalize = list()
    imagesList = list()
    for _ in caseFile["retrainList"]:
        candidateImage = max(closestClusterDist, key=closestClusterDist.get)
        E = closestClusterDist[candidateImage]
        closestClusterDist[candidateImage] = 1e9
        if len(unsafeImages) < totalUc:
            imagesList.append(candidateImage)
            toNormalize.append(E)
            totalImages.append(candidateImage)
            unsafeImages.append(candidateImage)
        else:
            break
    probList = list()
    probList2 = list()
    probList3 = list()
    for i, val in enumerate(toNormalize):
        probList.append(1 - (val/sum(toNormalize)))
    for i, val in enumerate(probList):
        if i == 0:
            offset = 0
        else:
            offset = probList2[i-1]
        probList2.append(val + offset)
    for i, val in enumerate(probList2):
        if (max(probList2) - min(probList2)) == 0:
            print(unsafeImages, toNormalize, probList, probList2)
        probList3.append((val - min(probList2)) / (max(probList2) - min(probList2)))
    while len(unsafeImages) < Ub:
        randNum = random.uniform(0, 1)
        for z in range(0, len(probList)):
            if randNum < probList[z]:
                totalImages.append(imagesList[z])
                unsafeImages.append(imagesList[z])
    return totalImages


def generateHMEntropy():
    #print("Entropy-Bagging")
    global clsWithAssImages
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3)
    #print("Ub", math.ceil(Ub))
    totalImages = []
    a = 0
    imagesEntropy = torch.load(join(caseFile["improveRCCDists"], "imagesEntropy.pt"))
    for clusterID in clsWithAssImages['clusters']:
        a += 1
        if 'assigned' in clsWithAssImages['clusters'][clusterID]:
            unsafeImages = []
            clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
            if clustLen > 1:
                imagesList = list()
                toNormalize = list()
                maxi = 0
                while maxi > -1e9:
                    candidateImage = max(imagesEntropy.keys(), key=(lambda k: imagesEntropy[k]))
                    maxi = imagesEntropy[candidateImage]
                    imagesEntropy[candidateImage] = -1e9
                    if len(unsafeImages) < clusterUCs[clusterID]:
                        if len(unsafeImages) < Ub:
                            imagesList.append(candidateImage)
                            toNormalize.append(maxi)
                            totalImages.append(candidateImage)
                            unsafeImages.append(candidateImage)
                if len(imagesList) > 1:
                    probList = list()
                    probList2 = list()
                    probList3 = list()
                    for i, val in enumerate(toNormalize):
                        probList.append(1 - (val/sum(toNormalize)))
                    for i, val in enumerate(probList):
                        if i == 0:
                            offset = 0
                        else:
                            offset = probList2[i-1]
                        probList2.append(val + offset)
                    for i, val in enumerate(probList2):
                        if (max(probList2) - min(probList2)) == 0:
                            print(unsafeImages, toNormalize, probList, probList2)
                        probList3.append((val - min(probList2)) / (max(probList2) - min(probList2)))
                    while len(unsafeImages) < Ub:
                        randNum = random.uniform(0, 1)
                        for z in range(0, len(probList)):
                            if randNum < probList[z]:
                                if len(unsafeImages) < Ub:
                                    totalImages.append(imagesList[z])
                                    unsafeImages.append(imagesList[z])
                else:
                    while len(unsafeImages) < Ub:
                        totalImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][0])
                        unsafeImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][0])
    totalImages = totalImages[0:math.ceil(totalUb)]
    #print("UnsafeSet:", len(totalImages))
    return totalImages

def generateTestSet(caseFile, bagPath):
    clsWithAssImages = torch.load(caseFile["clsPath"])
    csvPath = caseFile["testCSV"]
    retrainData = caseFile["testDataSet"]
    imgClasses = retrainData.dataset.classes
    imgListX = {}
    imageList = pd.read_csv(csvPath, names=["image", "result", "expected", "predicted"].append(imgClasses))
    for index, row in imageList.iterrows():
        imgListX[row["image"]] = 0
    totalImages = []
    copyImages = []
    maxUc = 0
    for clusterID in clsWithAssImages['clusters']:
        if len(clsWithAssImages['clusters'][clusterID]['members']) > maxUc:
            maxUc = len(clsWithAssImages['clusters'][clusterID]['members'])
    for clusterID in clsWithAssImages['clusters']:
        unsafeImages = []
        i = 0
        for img in clsWithAssImages['clusters'][clusterID]['members']:
            unsafeImages.append(img)
            totalImages.append(img)
            i += 1
        while len(unsafeImages) < maxUc:
            index = random.randint(0, len(unsafeImages) - 1)
            unsafeImages.append(unsafeImages[index])
            totalImages.append(unsafeImages[index])
            copyImages.append(unsafeImages[index])
    dupCount = 0
    imgExt = caseFile["imgExt"]
    for img in copyImages:
        testImage = basename(img)
        imgName = str(testImage.split("_")[1])
        imgClass = str(testImage.split("_")[2])
        if len(imgName.split("_")) > 2:
            imgName = imgName.split("_")[1] + "_" + imgName.split("_")[2]
            imgClass = str(testImage.split("_")[3])
        dstDir = join(bagPath, imgClass)
        if not exists(dstDir):
            os.makedirs(dstDir)
        dstFile = join(dstDir, testImage)
        if exists(dstFile):
            dstFile = join(dstDir, imgName + "_" + str(dupCount) + imgExt)
            dupCount += 1
        if testImage.split("_")[0] == "Test":
            srcFile = join(caseFile["testDataPath"], imgClass, imgName + imgExt)
        else:
            srcFile = join(caseFile["trainDataPath"], imgClass, imgName + imgExt)
        if srcFile in imgListX:
            del imgListX[srcFile]
        shutil.copy(srcFile, dstFile)
    #for srcFile in imgListX:
    #    dstDir = join(bagPath, basename(dirname(srcFile)))
    #    if not exists(dstDir):
    #        os.makedirs(dstDir)
    #    dstFile = join(dstDir, basename(srcFile))
    #    if exists(dstFile):
    #        dstFile = join(dstDir, str((basename(srcFile)).split(".")[0]) + "_" + str(dupCount) + imgExt)
    #        dupCount += 1
    #    shutil.copy(srcFile, dstFile)

def generateSEDE(clsWithAssImages):
    eval_imgs = 50
    totalImages = []
    for cID in clsWithAssImages['clusters']:
        print(cID)
        cFile = join(caseFile["filesPath"], "GeneratedImages", str(cID), "config.pt")
        csvPath = join(caseFile["filesPath"], "GeneratedImages", str(cID), "results.csv")
        if isfile(cFile) and isfile(csvPath):
            cPART = torch.load(cFile)
        else:
            continue
        imageList = pd.read_csv(csvPath)
        #paramNameList = ["cam_dir0", "cam_dir1", "cam_dir2", "cam_loc0", "cam_loc1", "cam_loc2", "lamp_loc0", "lamp_loc1",
        #                 "lamp_loc2", "lamp_col0", "lamp_col1", "lamp_col2", "lamp_dir0", "lamp_dir1", "lamp_dir2",
        #                 "lamp_eng", "head_pose0", "head_pose1", "head_pose2", "pose"] #IEE V1
        paramNameList = ["head0", "head1", "head2", "lampcol0", "lampcol1",
                  "lampcol2", "lamploc0", "lamploc1", "lamploc2", "lampdir0", "lampdir1", "lampdir2", "cam",
                  "age", "hue", "iris", "sat", "val", "freckle", "oil", "veins", "eyecol", "gender"] #IEE V2
        paramDict = {}
        for j in range(0, nVar):
            paramDict[paramNameList[j]] = []
        for index, row in imageList.iterrows():
            if not row["DNNResult"]:
                for i in range(0, nVar):
                    paramDict[paramNameList[i]].append(float(row[paramNameList[i]]))

        caseFile["SimDataPath"] = join(caseFile["filesPath"], "Pool")
        outDir = join(caseFile["filesPath"], "Evaluation", str(cID))
        if not exists(outDir):
            makedirs(outDir)
        n = 0
        for j in range(0, len(cPART['rules'])):
            toEval = int(cPART['portions'][j] * int(eval_imgs))
            total = 0
            while total < toEval:
                print(eval_imgs - total, end="\r")
                #x = setX(1, "R")
                x = setX_2(1, "R")
                #x = setNewX(x, paramDict, paramNameList, cPART['rules'][j], cPART['val1x'][j], cPART['val2x'][j])
                #imgPath, F = generateAnImage(x, caseFile)
                imgPath, F = generateHuman(x, caseFile)
                if F:
                    imgPath += ".png"
                    N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, None)
                    # DNNResult2, pred = testModelForImg(caseFile["DNN2"], L, imgPath, caseFile)
                    if DNNResult:
                        n += 1
                    # if DNNResult2:
                    #    n2 += 1
                    total += 1
                    totalImages.append(join(dirname(imgPath), L, basename(imgPath)))
    return totalImages
def generateSmartBaggingHM(imgList):
    print("HM-Bagging")
    global clsWithAssImages
    #print("SmartBagging")
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 1) #U4/U5
    #clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.1) #U7/U8
    if len(imgList) == 0:
        totalImages = []
        a = 0
        for clusterID in clsWithAssImages['clusters']:
            a += 1
            closestClusterName = torch.load(join(caseFile["improveRCCDists"], "closestClusterName.pt"))
            closestClusterDist = torch.load(join(caseFile["improveRCCDists"], "closestClusterDist.pt"))
            if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                unsafeImages = []
                clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                if clustLen > 1:
    #                Uc = clusterUCs[clusterID]
    #                i = 0
    #                while len(unsafeImages) < Uc:
    #                    if i >= len(clsWithAssImages['clusters'][clusterID]['assigned']):
    #                        i = 0
    #                    unsafeImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][i])
    #                    totalImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][i])
    #                i = 0
    #                while len(unsafeImages) < Ub:
    #                    if i >= len(clsWithAssImages['clusters'][clusterID]['assigned']):
    #                        i = 0
    #                    unsafeImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][i])
    #                    totalImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][i])
    #                    i += 1
                    toNormalize = list()
                    imagesList = list()
                    breakFlag = False
                    for order in range(0, len(clsWithAssImages['clusters'])):
                        print(str(int(100.00*(a/len(clsWithAssImages['clusters'])))) + "%",
                              str(int(100.00*(order/len(clsWithAssImages['clusters'])))) + "%", end="\r")
                        for _ in caseFile["retrainList"]:
                            candidateImage = min(closestClusterDist[order].keys(),
                                                 key=(lambda k: closestClusterDist[order][k]))
                            candidateClusterID = closestClusterName[order][candidateImage]
                            dif = closestClusterDist[order][candidateImage]
                            closestClusterDist[order][candidateImage] = 1e9
                            if candidateClusterID == clusterID:
                                if len(unsafeImages) < clusterUCs[clusterID]:
                                        imagesList.append(candidateImage)
                                        toNormalize.append(dif)
                                        totalImages.append(candidateImage)
                                        unsafeImages.append(candidateImage)
                                else:
                                    breakFlag = True
                            if breakFlag:
                                break
                    #if False:
                    if len(imagesList) > 1:
                        probList = list()
                        probList2 = list()
                        probList3 = list()
                        for i, val in enumerate(toNormalize):
                            probList.append(1 - (val/sum(toNormalize)))
                        for i, val in enumerate(probList):
                            if i == 0:
                                offset = 0
                            else:
                                offset = probList2[i-1]
                            probList2.append(val + offset)
                        for i, val in enumerate(probList2):
                            if (max(probList2) - min(probList2)) == 0:
                                print(unsafeImages, toNormalize, probList, probList2)
                            probList3.append((val - min(probList2)) / (max(probList2) - min(probList2)))
                        while len(unsafeImages) < Ub:
                            randNum = random.uniform(0, 1)
                            for z in range(0, len(probList)):
                                if randNum < probList[z]:
                                    totalImages.append(imagesList[z])
                                    unsafeImages.append(imagesList[z])
                    else:
                        while len(unsafeImages) < Ub:
                            totalImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][0])
                            unsafeImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][0])
            #print(clusterID, len(unsafeImages), len(totalImages))
    else:
        totalImages = imgList
    totalImages = totalImages[0:math.ceil(totalUb)]
    return totalImages


def generateRandomBagging():
    global clsWithAssImages
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3) #U4/U5
    #clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.1) #U7/U8
    totalImages = []
    #print("RandomBagging")
    for clusterID in clsWithAssImages['clusters']:
        if 'assigned' in clsWithAssImages['clusters'][clusterID]:
            unsafeImages = []
            clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
            if clustLen > 0:
                Uc = clusterUCs[clusterID]
                i = 0
                while len(unsafeImages) < Uc:
                    unsafeImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][i])
                    totalImages.append(clsWithAssImages['clusters'][clusterID]['assigned'][i])
                    i += 1
                while len(unsafeImages) < Ub:
                    index = random.randint(0, len(unsafeImages) - 1)
                    unsafeImages.append(unsafeImages[index])
                    totalImages.append(unsafeImages[index])
    totalImages = totalImages[0:totalUb]
    return totalImages


def generateBL2():
    totalImages = []
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3)
    improvFiles = caseFile["retrainList"]
    while len(totalImages) < math.ceil(totalUc):
        index = random.randint(0, len(improvFiles) - 1)
        totalImages.append(improvFiles[index])
    while len(totalImages) < math.ceil(totalUb):
        index = random.randint(0, len(totalImages) - 1)
        totalImages.append(totalImages[index])
    totalImages = totalImages[0:math.ceil(totalUb)]
    return totalImages


def generateBL1():
    totalImages = []
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3)
    net = loadDNN(caseFile, caseFile["modelPath"])
    net = net.eval()
    improvFiles = caseFile["retrainList"]
    improvFiles = improvFiles[0:math.ceil(totalUc)]
    dumbImages = []
    while len(dumbImages) < totalUc:
        if len(totalImages) < math.ceil(totalUb):
            imgID = random.randint(0, len(improvFiles) - 1)
            dumbImages.append(improvFiles[imgID])
            fileClass = basename(dirname(improvFiles[imgID]))
            if not (testModule.testModelForImg(net, fileClass, improvFiles[imgID], caseFile)):
                totalImages.append(improvFiles[imgID])
    print("Total Failing:", len(totalImages))
    while len(totalImages) < math.ceil(totalUb):
        index = random.randint(0, len(totalImages) - 1)
        totalImages.append(totalImages[index])
    totalImages = totalImages[0:math.ceil(totalUb)]
    return totalImages

def generateBLE():
    clusterUCs, totalAssigned, totalUc, totalUb, Ub = getUCs(caseFile, 0.3)
    imagesEntropy = torch.load(join(caseFile["improveRCCDists"], "imagesEntropy.pt"))
    entropyList = list()
    for _ in imagesEntropy:
        candidateImage = max(imagesEntropy.keys(), key=(lambda k: imagesEntropy[k]))
        imagesEntropy[candidateImage] = 0
        entropyList.append(candidateImage)
    totalImages = entropyList[0:math.ceil(totalUc)]
    while len(totalImages) < math.ceil(totalUb):
        index = random.randint(0, len(totalImages) - 1)
        totalImages.append(totalImages[index])
    totalImages = totalImages[0:math.ceil(totalUb)]
    return totalImages

def collectData(bagPath, imagesList):
    global caseFile
    global clsWithAssImages
    net = loadDNN(caseFile, caseFile["modelPath"])
    net = net.eval()
    mode = caseFile["retrainMode"]
    clsPath = caseFile["assignPTFile"]
    if imagesList is None:
        imgLst = list()
    else:
        imgLst = imagesList
    totalImages = []
    HUDDUnsafeSet = {}

    HUDDPath = join(caseFile["filesPath"], "UnsafeSet.pt")
    if mode.startswith("SEDE"):

        clsWithAssImages = torch.load(join(caseFile["filesPath"], "ClusterAnalysis_" + str(caseFile["clustMode"]),
                                      caseFile["selectedLayer"] + ".pt"))
        totalImages = generateSEDE(clsWithAssImages)
    else:
        clsWithAssImages = torch.load(clsPath)
        if mode.startswith("HUDD"):
            #if isfile(HUDDPath):
            #    HUDDUnsafeSet = torch.load(HUDDPath)
            #    totalImages = HUDDUnsafeSet["UnsafeSet"]
            #else:
            if mode.endswith("E"):
                #totalImages = generateSmartBaggingEntropy()
                totalImages = generateHMEntropy()
            else:
                totalImages = generateSmartBaggingHM(imgLst)
                #totalImages = generateHMEntropy()
            HUDDUnsafeSet["UnsafeSet"] = totalImages
                #torch.save(HUDDUnsafeSet, HUDDPath)
            #totalImages = generateRandomBagging()
        if mode == "BL1":
            totalImages = generateBL1()
        if mode == "BL2":
            totalImages = generateBL2()
        if mode == "BLE":
            totalImages = generateBLE()
    dupCount = 0
    failCount = 0
    failCount2 = 0
    print("Img Length;", len(totalImages))
    for img in totalImages:
        trainImage = basename(img)
        imgExt = caseFile["imgExt"]
        imgName = str(trainImage.split(".")[0])
        imgClass = str(basename(dirname(img)))
        if len(imgName.split("_")) > 2:
            imgName = imgName.split("_")[1] + "_" + imgName.split("_")[2]
        dstDir = join(bagPath, imgClass)
        if not exists(dstDir):
            os.makedirs(dstDir)
        dstFile = join(dstDir, trainImage)
        if exists(dstFile):
            dstFile = join(dstDir, imgName + "_" + str(dupCount) + imgExt)
            dupCount += 1
        if not (testModule.testModelForImg(net, imgClass, img, caseFile)):
        #if not (testModule.testModelForImg(net, imgClass, join(dirname(dirname(img)), basename(img)), caseFile)):
            failCount2 += totalImages.count(img)
            failCount += 1/totalImages.count(img)
        srcFile = join(caseFile["improveDataPath"], imgClass, imgName + imgExt)
        #srcFile = join(dirname(dirname(img)), basename(img))
        imgLst.append(img)
        shutil.copy(srcFile, dstFile)
    caseFile[mode]["totalFailCount"] += failCount2
    caseFile[mode]["failCount"] += failCount
    caseFile[mode]["dupCount"] += dupCount
    caseFile[mode]["BaggedUnsafeSetSize"] = len(totalImages)
    return imgLst

def BL4_Data(improvSet, bagPath, net, datasetName):
    net = net.eval()
    transformer = setupTransformer(datasetName)
    _, impovmentDataset, _ = testModule.loadData(improvSet, datasetName, 4, 64, None, None)

    improvFiles = list()
    for src_dir, dirs, files in os.walk(improvSet):
        for file_ in files:
            if (file_.endswith(".jpg")) or (file_.endswith(".png")) or (file_.endswith(".ppm")):
                improvFiles.append(join(src_dir, file_))

    unsafeImages = []
    imgLst = list()

    selectionSize = len(improvFiles)
    print("Total Images to be selected Randomly from IS:", selectionSize)

    for img in improvFiles:
        fileName = basename(img)
        fileClass = basename(dirname(img))
        image = Image.open(join(improvSet, fileClass, fileName))
        imageTensor = transformer(image).float()
        imageTensor = imageTensor.unsqueeze_(0)
        imageTensor = Variable(imageTensor, requires_grad=False)
        imageTensor.detach()
        if not (testModule.testModelForImg(net, imageTensor, impovmentDataset.dataset.classes.index(fileClass))):
            unsafeImages.append(img)

    print("Total Failing images found: ", str(len(unsafeImages)))

    dupCount = 0
    for image in unsafeImages:
        srcFileName = basename(image)
        srcFileClass = basename(dirname(image))
        imgLst.append(image)
        dstFile = join(bagPath, srcFileClass, srcFileName)
        if not exists(bagPath):
            os.makedirs(bagPath)
        if not exists(join(bagPath, srcFileClass)):
            os.makedirs(join(bagPath, srcFileClass))
        if exists(dstFile):
            newFileName = str(srcFileName.split(".")[0]) + "_" + str(dupCount) + "." + str(srcFileName.split(".")[1])
            dstFile = join(bagPath, srcFileClass, newFileName)
            dupCount += 1
        shutil.copy(image, dstFile)

    print("Total Duplicated Images:", str(dupCount))
    print("Total Retraining Images is ", str(len(unsafeImages)))

    return imgLst


def process_json_list(json_list, img):
    ldmks = [eval(s) for s in json_list]
    return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])


def computeAngle(data, img):
    ldmks_iris = process_json_list(data['iris_2d'], img)
    look_vec = list(eval(data['eye_details']['look_vec']))
    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    look_vec[1] = -look_vec[1]
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))
    angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
                                                                                      point_B[1] - point_A[1])
    angle = (angle * 180) / math.pi
    while (angle < 0):
        angle = angle + 360
    return angle, point_A, point_B, point_C


def getclass(file, img_dir, DS):
    fileName = str(file).split(".")[0]
    img = cv2.imread(join(img_dir, fileName + ".jpg"))
    json_fn = join(img_dir, fileName + ".json")
    data_file = open(json_fn)
    data = json.load(data_file)

    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    ldmk1 = ldmks_interior_margin[4]
    ldmk2 = ldmks_interior_margin[12]
    x1 = int(ldmk1[0])
    y1 = int(ldmk1[1])
    x2 = int(ldmk2[0])
    y2 = int(ldmk2[1])

    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist = int(dist)
    angle, point_A, point_B, point_C = computeAngle(data, img)
    if (DS == 'OC'):
        if dist < 20:
            classe = "Closed"
        else:
            classe = "Opened"

    else:
        if 0 <= angle < 22.5:
            classe = "MiddleLeft"
        if 22.5 < angle < 67.5:
            classe = "TopLeft"
        if 67.5 < angle < 112.5:
            classe = "TopCenter"
        if 112.5 < angle < 157.5:
            classe = "TopRight"
        if 157.5 < angle < 202.5:
            classe = "MiddleRight"
        if 202.5 < angle < 247.5:
            classe = "BottomRight"
        if 247.5 < angle < 292.5:
            classe = "BottomCenter"
        if 292.5 < angle < 337.5:
            classe = "BottomLeft"
        if angle >= 337.5:
            classe = "MiddleLeft"

    return classe



def getImgLst(path, imgList, prefix):
    if imgList is None:
        imgList = list()
    for path, subdirs, files in os.walk(path):
        for filename in files:
            testPath = path + "/" + filename
            srcFileClass = testPath.split(os.sep)[len(testPath.split(os.sep)) - 2]
            imgList.append(prefix + srcFileClass + "/" + filename)
    return imgList


def createData(srcDir, dstDir, weightPath):
    to_train_data(srcDir, dstDir, None, weightPath)

def labelImages(path):
    counter = 1
    for src_dir, dirs, files_ in os.walk(path):
        for file in files_:
            if file.endswith(".png"):
                imgPath = join(src_dir, file)
                npPath = join(src_dir, file.split(".png")[0] + ".npy")
                margin1 = 10.0
                margin2 = -10.0
                margin3 = 10.0
                margin4 = -10.0
                configFile = np.load(npPath, allow_pickle=True)
                #print(configFile)
                #print(configFile.item())
                configFile = configFile.item()
                #print(configFile['config']['head_pose'])
                #print(configFile['config'])
                HP1 = configFile['config']['head_pose'][0]
                HP2 = configFile['config']['head_pose'][1]
                #print(configFile)
                originalDst = None
                if HP1 > margin1:
                    if HP2 > margin3:
                        originalDst = "BottomRight"
                    elif HP2 < margin4:
                        originalDst = "BottomLeft"
                    elif margin4 <= HP2 <= margin3:
                        originalDst = "BottomCenter"
                elif HP1 < margin2:
                    if HP2 > margin3:
                        originalDst = "TopRight"
                    elif HP2 < margin4:
                        originalDst = "TopLeft"
                    elif margin4 <= HP2 <= margin3:
                        originalDst = "TopCenter"
                elif margin2 <= HP1 <= margin1:
                    if HP2 > margin3:
                        originalDst = "MiddleRight"
                    elif HP2 < margin4:
                        originalDst = "MiddleLeft"
                    elif margin4 <= HP2 <= margin3:
                        originalDst = "MiddleCenter"
                originalDst = join(path, originalDst)
                file_name = join(originalDst, str(counter) + ".png")
                if not exists(originalDst):
                    os.mkdir(originalDst)
                shutil.move(imgPath, file_name)
                counter += 1
def to_train_data(src, dstDir, evidence_path, weightPath):
    all_pngs = get_all_pngs(src)
    face_detector = get_face_detector(weightPath)

    labels_data = []
    imgs_data = []
    ori_imgs_file = []
    file_num = len(all_pngs)
    missingLabels = 0
    print(src)
    print(file_num)
    for idx, apng in enumerate(all_pngs):
        print("process: ", apng, idx, "/", file_num)
        label_file = apng.split(".png")[0] + ".npy"
        if not exists(label_file):
            missingLabels = missingLabels + 1
            continue
        label_value = get_label(label_file)
        img_value = get_img(apng)
        faces = face_detector(img_value, 1)
        if len(faces) != 1:
            print("face_detector finds ", len(faces), " faces in: ", apng, " ..skip..")
            continue

        new_img_value, new_label_value = crop_img_lab(idx, faces, img_value, label_value, evidence_path)

        labels_data.append(new_label_value)
        imgs_data.append(new_img_value)
        ori_imgs_file.append(apng)

    labels_data = np.concatenate(labels_data, axis=0)
    imgs_data = np.concatenate(imgs_data, axis=0)
    print("labels_data shape: ", labels_data.shape, "imgs_data shape: ", imgs_data.shape)
    print("Missing Labels Count:" + str(missingLabels))
    dataset = {}
    dataset["Image"] = imgs_data
    dataset["Label"] = labels_data
    dataset["Origin"] = ori_imgs_file
    np.save(dstDir, dataset)



def getUCs(caseFile, factor=0.3):
    datasetName = caseFile["datasetName"]
    retrainLength = len(caseFile["retrainList"])
    trainFlag = caseFile["trainFlag"]
    #testTotal = caseFile["testTotal"]
    testTotal = 2825
    testTotal = 3000
    #trainTotal = caseFile["trainTotal"]
    #trainError = caseFile["trainError"]
    #testError = caseFile["testError"]

    if isfile(caseFile["assignPTFile"]):
        clsWithAssImages = torch.load(caseFile["assignPTFile"])
    else:
        clsWithAssImages = torch.load(caseFile["clsPath"])
    clusterMembers = {}
    totalAssigned = 0
    for clusterID in clsWithAssImages['clusters']:
        if len(clsWithAssImages['clusters'][clusterID]['members']) > 1:
            testSize = 0
            trainSize = 0
            clusterMembers[clusterID] = {}
            if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                totalAssigned += clustLen
            for member in clsWithAssImages['clusters'][clusterID]['members']:
                if member.startswith("Test_"):
                    testSize += 1
                elif member.startswith("Train_"):
                    trainSize += 1
            clusterMembers[clusterID]["trainSize"] = trainSize
            clusterMembers[clusterID]["testSize"] = testSize
    clusterUc = {}
    totalUc = 0
    maxUc = 0
    for clusterID in clusterMembers:
        clusterUc[clusterID] = {}
        Uc5 = clusterMembers[clusterID]["testSize"] * factor
        Uc1 = (clusterMembers[clusterID]["testSize"]/testTotal) * retrainLength
        Uc6 = Uc1 / 4
        Uc3 = clusterMembers[clusterID]["testSize"] * 0.25
        Uc = Uc5
        if trainFlag:
            clusterUc[clusterID] = Uc
        #    clusterUc[clusterID] = U2
        else:
            clusterUc[clusterID] = Uc
        #    clusterUc[clusterID] = U3
        clusterUc[clusterID] = math.ceil(clusterUc[clusterID])
        totalUc += clusterUc[clusterID]

        if clusterUc[clusterID] > maxUc:
            maxUc = clusterUc[clusterID]

    #if caseFile["datasetName"].startswith("IEEKP"):
    #    totalUb = (trainTotal/3)/(len(caseFile["components"])-1)
    #    if len(clusterMembers) > 0:
    #        maxUc = (trainTotal/3)/(len(caseFile["components"])-1)/(len(clusterMembers))
    #    else:
    #        maxUc = 0
    #else:
    #    totalUb = (trainTotal/3)
    totalAssigned = totalUc
    totalUb = maxUc * len(clusterMembers)
    #totalAssigned = (float(testError)/float(testTotal)) * retrainLength
    #print(totalAssigned)
    #print(totalUb)
    return clusterUc, totalAssigned, totalUc, totalUb, maxUc

def loadDNN(caseFile, modelPath):
    if modelPath is None:
        modelPath = caseFile["modelPath"]
    Alex = caseFile["Alex"]
    KP = caseFile["KP"]
    datasetName = caseFile["datasetName"]
    numClass = caseFile["numClass"]
    scratchFlag = caseFile["scratchFlag"]
    if Alex:
        if datasetName.startswith("HPD"):
            net = dnnModels.AlexNetIEE(numClass)
        else:
            net = dnnModels.AlexNet(numClass)
    elif KP:
        net = dnnModels.KPNet()
    if torch.cuda.is_available():
        if not scratchFlag:
            weights = torch.load(modelPath)
            if Alex:
                net.load_state_dict(weights)
            elif KP:
                net.load_state_dict(weights.state_dict())
        net = net.to('cuda')
        net.cuda()
        net.eval()
        DNN = net
    else:
        if not scratchFlag:
            weights = torch.load(modelPath, map_location=torch.device('cpu'))
            if Alex:
                net.load_state_dict(weights)
            elif KP:
                net.load_state_dict(weights.state_dict())
        net.eval()
        DNN = net
    return DNN




def get_data(f_path, max_num=0, shuffle=True):
    dataset = np.load(f_path, allow_pickle=True)
    dataset = dataset.item()

    x_data = dataset["data"]
    if max_num>0:
        if shuffle:
            r_idx = np.random.permutation(x_data.shape[0])
            r_idx = r_idx[:max_num]
            x_data = x_data[r_idx]
        else:
            x_data = x_data[:max_num]

    #x_data = x_data[:max_num]
    x_data = x_data.astype(np.float32)
    x_data = x_data / 255.
    #x_data = x_data.reshape((-1,1,x_data.shape[-2], x_data.shape[-1]))
    x_data = x_data[:,np.newaxis]
    #print("x_data shape: ", x_data.shape)

    y_data = dataset["label"]
    if max_num>0:
        if shuffle:
            y_data = y_data[r_idx]
        else:
            y_data = y_data[:max_num]
    y_data = y_data.astype(np.float32)


    return x_data, y_data


class KPDataset(Dataset):
    def __init__(self, x_data, y_data, using_gm=True, gm_sigma=5.0, gm_scale=10.0, transforms=None):
        self.x_data = x_data
        self.y_data = y_data

        self.using_gm = using_gm
        self.transforms = transforms
        self.gm_sigma = gm_sigma
        self.gm_scale = gm_scale


    def __getitem__(self, index):
        img = self.x_data[index]
        kps = self.y_data[index]
        if self.transforms:
            img = self.transforms(img)

        if self.using_gm:
            gm = GaussianMap(kps, img.shape[-2], img.shape[-1], self.gm_sigma)
            (gm_kps, msk_kps) = gm.create_heatmaps(self.gm_scale)
            kps = {"kps": kps, "gm": torch.from_numpy(gm_kps)} #"msk" : torch.from_numpy(msk_kps)
            #print("gm_kps: ", gm_kps.shape)
        return img, kps

    def __len__(self):
        return len(self.x_data)

class ToTensor(object):
    def __call__(self, img):
        # imagem numpy: C x H x W
        # imagem torch: C X H X W
        #img = img.transpose((0, 1, 2)) all the imgs are processed to 1 channel grayscale
        return torch.from_numpy(img)

class CloneArray(object):
    def __call__(self, img):
        img = img.repeat(3, axis=0)
        return img


class DataTransformer(object):
    def __init__(self):
        #self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        #self.imagenet_std  = np.array([0.229, 0.224, 0.225])
        return

    def tranform_basic(self):
        # some researchers prefer to convert a gray image to a psudo-rgb image, by introducing
        #     imagenet_mean and imangnet_std
        # In the IEE case, this trick does not improve accuracy. So we dont use this trick, but
        #     keeping the possibility here.

        #trans =  transforms.Compose([CloneArray(),
        #                             ToTensor(),
        #                             transforms.Normalize(self.imagenet_mean, self.imagenet_std)])
        trans =  transforms.Compose([ToTensor()])
        return trans

class DataIter(object):
    def __init__(self, dataset, valid_ratio, batch_size, pin_memory=False):
        self.pin_memory = pin_memory
        self.dataset = dataset
        self.num = len(dataset)
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        return

    def get_samplers(self, shuffle):
        indices = list(range(self.num))
        if shuffle:
            np.random.seed(data_random_seed)
            np.random.shuffle(indices)

        if self.valid_ratio > 0:
            split = int(np.floor(self.valid_ratio * self.num))
            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            return train_sampler, valid_sampler
        else:
            train_sampler = SubsetRandomSampler(indices)
            return train_sampler, None

    def get_iters(self, shuffle=False):
        if self.valid_ratio > 0:
            train_sampler,valid_sampler = self.get_samplers(shuffle)
            train_iter = DataLoader(self.dataset, batch_size=self.batch_size,
                sampler=train_sampler, pin_memory=self.pin_memory)
            valid_iter = DataLoader(self.dataset, batch_size=self.batch_size,
                sampler=valid_sampler, pin_memory=self.pin_memory)
            return train_iter, valid_iter
        else:
            train_iter = DataLoader(self.dataset, batch_size=self.batch_size,
                pin_memory=self.pin_memory, shuffle=shuffle)
            return train_iter, None


#
# Copyright (c) IEE 2019-2020.
# Created by Jun WANG, jun.wang@iee.lu, IEE, 2019.
#

import numpy as np

class GaussianMap(object):
    # landmarks.shape = [n,2], n = the number of landmarks
    def __init__(self, landmarks, width, height, sigma):
        self.landmarks = landmarks
        self.width = width
        self.height = height
        self.sigma = sigma

    def kernel(self, x, y):
        tmp_x = np.arange(0, self.width, 1, float)
        tmp_y = np.arange(0, self.height, 1, float)[:, np.newaxis]
        kn = np.exp(-((tmp_x - x) ** 2 + (tmp_y - y) ** 2) / (2 * self.sigma ** 2))
        return kn

    def create_heatmaps(self, gm_scale):
        y_data = []
        y_imp = []
        for lm in self.landmarks:
            gm = np.zeros((self.height, self.width), dtype=np.float32)
            msk = np.zeros((self.height, self.width), dtype=np.float32)
            if lm[0] > 0 and lm[1] > 0:
                gm = self.kernel(lm[0], lm[1])
                msk = msk + 1
            y_data.append(gm)
            y_imp.append(msk)
        y_data = np.array(y_data) * gm_scale
        y_imp = np.array(y_imp)
        # print("y_data: ", y_data.shape)
        return (y_data, y_imp)

def get_ave_xy(hmi, n_points=64, thresh=0):
    '''
    hmi      : heatmap np array of size (height,width)
    n_points : x,y coordinates corresponding to the top  densities to calculate average (x,y) coordinates


    convert heatmap to (x,y) coordinate
    x,y coordinates corresponding to the top  densities
    are used to calculate weighted average of (x,y) coordinates
    the weights are used using heatmap

    if the heatmap does not contain the probability >
    then we assume there is no predicted landmark, and
    x = -1 and y = -1 are recorded as predicted landmark.
    '''
    assert n_points > 1

    ind = hmi.argsort(axis=None)[-n_points:]  ## pick the largest n_points
    topind = np.unravel_index(ind, hmi.shape)
    index = np.unravel_index(hmi.argmax(), hmi.shape)
    i0, i1, hsum = 0, 0, 0
    for ind in zip(topind[0], topind[1]):
        h = hmi[ind[0], ind[1]]
        hsum += h
        i0 += ind[0] * h
        i1 += ind[1] * h

    i0 /= hsum
    i1 /= hsum
    if hsum / n_points <= thresh:
        i0, i1 = -1, -1
    return [i1, i0]

def transfer_xy_coord(hm, n_points=64, thresh=0.2):
    '''
    hm : np.array of shape (n-heatmap, height,width)

    transfer heatmap to (x,y) coordinates

    the output contains np.array (Nlandmark, 2)
    * 2 for x and y coordinates, containing the landmark location.
    '''
    assert len(hm.shape) == 3
    Nlandmark = hm.shape[0]
    # est_xy = -1*np.ones(shape = (Nlandmark, 2))
    est_xy = []
    for i in range(Nlandmark):
        hmi = hm[i, :, :]
        est_xy.append(get_ave_xy(hmi, n_points, thresh))
    return est_xy  ## (Nlandmark, 2)


def transfer_target(y_pred, thresh=0, n_points=64):
    '''
    y_pred : np.array of the shape (N, Nlandmark, height, width)

    output : (N, Nlandmark, 2)
    '''
    y_pred_xy = []
    for i in range(y_pred.shape[0]):
        hm = y_pred[i]
        y_pred_xy.append(transfer_xy_coord(hm, n_points, thresh))

    return (np.array(y_pred_xy))


def HUDDevaluate_Pop2_A(X, caseFile, Y):
    print("POP2")
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        j = len(fy)
        if x is None:
            fy.append(math.inf)
            continue
        imgPath, F = generateAnImage(x, caseFile)
        imgPath += ".png"
        fy_ = math.inf
        f1 = math.inf
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)
            f1 = D / CR
            if f1 <= 1:
                if not DNNResult:
                    x2 = Y[j]
                    imgPath2, F2 = generateAnImage(x2.X, caseFile)
                    imgPath2 += ".png"
                    N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH)
                    _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer])
                    fp = doParamDist(x, x2.X, xl, xu)
                    fy_ = fp
                else:
                    fy_ = 2 - N
            else:
                fy_ = 2 + f1
        f1_.append(f1)
        fy.append([fy_])
    print("F1:", f1_)
    return fy




def HUDDevaluate2(x, caseFile, prevEN, prevPOP):
    global indvdSize
    f1l = []
    f2_l = []
    fyl = []
    indiv = []
    N_ = []
    PS = 0
    problemDict = {processX(x): {}}
    for i in range(0, indvdSize):
        x1 = getI(x, i)
        indiv.append(x1)
        X1 = processX(x1)
        Y1 = None
        if prevPOP is not None:
            Y1 = getI(prevPOP.X, i)
        problemDict2 = HUDDevaluate(x1, caseFile, prevEN, Y1)
        problemDict[X1] = problemDict2
        if problemDict2["Face"]:
            N_.append(problemDict2["Adjusted_N"])
            f1l.append(problemDict2["F1"])
            f2_l.append(problemDict2["F2_"])
            if problemDict2["DNNResult"] and problemDict2["Low_N"]:
                PS += 1
        fyl.append(problemDict2["FY_"])
    ANPD = getANPD(indiv, caseFile, True)
    MP = f2_l.count(0) / len(f2_l)
    if PS == 0:
        PS = 1
    else:
        PS = 1
    IP = 0
    for z in f1l:
        if z <= 1.0:
            IP += 1
    IP = IP / len(f1l)
    if prevEN is not None and (len(f1l) == indvdSize):
        if float(max(f1l)) <= 1.0 and float(max(f2_l)) == 0.0:
            f3 = [1 - ANPD]
        else:
            if ANPD == 0 or IP == 0 or MP == 0:
                f3 = [math.inf]
            else:
                f3 = [1 + ((1 / (IP)) * (1 / MP) * math.log10(1 / ANPD))]
    else:
        f3 = [math.inf]
    FY = [i * PS for i in fyl]
    for any_ in problemDict:
        problemDict[any_]["F3"] = f3
        problemDict[any_]["FY"] = FY
        problemDict[any_]["ANPD"] = ANPD

    print("F1:", max(f1l))
    print("FY:", FY)
    print("%:", str(100 * MP)[0:5] + "%")
    print("F3:", f3)
    print("IP:", str(100 * IP)[0:5] + "%")
    problemDict[processX(x)]["N"] = N_
    # print("Individual -- F3 = ", f3)
    # print("Individual -- Fy = ", fyl)
    return problemDict


def evalImages(clsData, caseFile):
    global layer
    for member in clsData['clusters'][clusterID]['members']:
        fileName = member.split("_")[1]
        data = np.load(caseFile["testDataNpy"], allow_pickle=True)
        data = data.item()
        configData = data['config'][int(fileName) - 1]
        x = [configData['cam_look_direction'][0], configData['cam_look_direction'][1],
             configData['cam_look_direction'][2], configData['cam_loc'][0], configData['cam_loc'][1],
             configData['cam_loc'][2], configData['lamp_loc_Lamp'][0], configData['lamp_loc_Lamp'][1],
             configData['lamp_loc_Lamp'][2], configData['lamp_color_Lamp'][0], configData['lamp_color_Lamp'][1],
             configData['lamp_color_Lamp'][2], configData['lamp_direct_xyz_Lamp'][0],
             configData['lamp_direct_xyz_Lamp'][1], configData['lamp_direct_xyz_Lamp'][2],
             configData['lamp_energy_Lamp'], configData["head_pose"][0], configData["head_pose"][1],
             configData["head_pose"][2]]
        imgPath, _ = generateAnImage(x, caseFile)
        processImage(imgPath + ".png", join(caseFile["outputPath"],
                                            "IEEPackage/clsdata/mmod_human_face_detector.dat"))
        layersHM, entropy = generateHeatMap(imgPath + ".png", caseFile["DNN"],
                                            caseFile["datasetName"],
                                            caseFile["outputPath"], False, None, None,
                                            caseFile["imgExt"], None)
        newImgPath = join(caseFile["DataSetsPath"], "TestSet_Backup", str(int(fileName) - 1) + ".png")
        print()
        layersHM2, entropy = generateHeatMap(newImgPath, caseFile["DNN"],
                                             caseFile["datasetName"],
                                             caseFile["outputPath"], False, None, None,
                                             caseFile["imgExt"], None)
        dist = doDistance(centroidHM,
                          layersHM[layer],
                          "Euc")

        dist2 = doDistance(centroidHM,
                           layersHM2[layer],
                           "Euc")
        print(1 - (clusterRadius / dist))
        print(1 - (clusterRadius / dist2))


def toCSV(member, outFile, CF, prevEN, prev_pop, probNum):
    ID = CF["ID"]
    counter = 1
    problemDict = HUDDevaluate(member.X, CF, prevEN, prev_pop)
    if probNum == 2:
        strW = "idx,clusterID"
        for _x_ in problemDict:
            if "Face" in problemDict[_x_]:
                if problemDict[_x_]["Face"]:
                    for nameX in problemDict[_x_]:
                        if nameX != "FY":
                            strW += "," + str(nameX)
                    break
        strW += ",cam_dir0,cam_dir1,cam_dir2,cam_loc0,cam_loc1,cam_loc2,lamp_loc0," \
                "lamp_loc1,lamp_loc2," \
                "head_pose0,head_pose1,head_pose2,pose,imgPath\r\n"
        outFile.writelines(strW)
    if probNum == 3:
        ID = 0
    for j in range(0, indvdSize):
        X = getI(member.X, j)
        X_ = processX(X)
        if problemDict[X_]["Face"]:
            if probNum == 2 or ((problemDict[X_]["DNNResult"]) and (probNum == 3)):
                imgPath = join(CF["filesPath"], "Pool", processX(X) + ".png")
                strMerge = str(counter) + "," + str(ID)
                for nameX in problemDict[X_]:
                    if nameX != "FY":
                        strMerge += "," + str(problemDict[X_][nameX])
                for j in range(0, len(X) - 1):
                    strMerge += "," + str(X[j])
                strMerge += "," + str(math.floor(X[len(X) - 1]))
                strMerge += "," + str(imgPath)
                strMerge += "\r\n"
                outFile.writelines(strMerge)
            counter += 1


if __name__ == '__main__':
    x_data, y_data = get_data(iee_train_data)
    print("train data: ", x_data.shape, y_data.shape)
    x_data, y_data = get_data(iee_test_data)
    print("test data: ", x_data.shape, y_data.shape)
    x_data, y_data = get_data(iee_real_data)
    print("real data: ", x_data.shape, y_data.shape)



