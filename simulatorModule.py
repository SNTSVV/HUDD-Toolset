#
# Copyright (c) IEE, University of Luxembourg 2021-2022.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2022.
#

import subprocess as sp
import pathlib as pl

from imports import np, random, makedirs, torch, pd, join, basename, exists, isfile, sys, os, cv2, dlib, dirname, \
    imageio, subprocess, shutil, math, random, distance, math, time, plt
from HeatmapModule import generateHeatMap, doDistance
from assignModule import getClusterData
from testModule import testModelForImg
import logging
import pickle

import config as cfg

components = cfg.components
blenderPath = cfg.blenderPath
nVar = cfg.nVar
indvdSize = cfg.indvdSize
BL = cfg.BL
width = cfg.width
height = cfg.height
globalCounter = random.randint(1, 999999999)


def getParamVals():  # IEE_V1
    param_list = ["cam_look_0", "cam_look_1", "cam_look_2", "cam_loc_0", "cam_loc_1", "cam_loc_2",
                  "lamp_loc_0", "lamp_loc_1", "lamp_loc_2", "head_0", "head_1", "head_2"]
    # TrainingSet parameters (min - max)
    cam_dirL = [-0.10, -4.67, -1.69]
    cam_dirU = [-0.08, -4.29, -1.27]
    cam_locL = [0.261, -5.351, 14.445]
    cam_locU = [0.293, -5.00, 14.869]  # constant
    lamp_locL = [0.361, -5.729, 16.54]
    lamp_locU = [0.381, -5.619, 16.64]  # constant

    #headL = [-41.86, -79.86, -64.30] # TrainingSet parameters
    #headU = [36.87, 75.13, 51.77] # TrainingSet parameters
    faceL = 8
    faceU = 8
    # TestSet parameters (min - max)
    headL = [-32.94, -88.10, -28.53]
    headU = [33.50, 74.17, 46.17]
    # fixing HP_2
    # headL = [-32.94, -88.10, -0.000001]
    # headU = [33.50, 74.17, 0]
    return param_list, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, headL, headU, faceL, faceU


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
    _, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, headL, headU, faceL, faceU = getParamVals()
    xl = []
    if ID == "L":
        for i in range(0, size):
            for c in cam_dirL:
                xl.append(c)
            for c in cam_locL:
                xl.append(c)
            for c in lamp_locL:
                xl.append(c)
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


def getI(x, i):
    return x[nVar * i:nVar * (i + 1)]


def doParamDist(x, y, xl, xu):
    x_ = []
    y_ = []
    for m in range(0, len(x)):
        if xu[m] == xl[m]:
            x_.append(0)
            y_.append(0)
        else:
            x_.append((x[m] - xl[m]) / (xu[m] - xl[m]))
            y_.append((y[m] - xl[m]) / (xu[m] - xl[m]))
    d = distance.cosine(x_, y_)
    if math.isnan(d):
        return 0
    return d


def getParams(mini, maxi, param, BL):
    param_1st = param[0]
    param_3rd = param[1]
    if param_1st < mini:
        param_1st = mini
    if param_1st > maxi:
        param_1st = maxi
    if param_3rd < mini:
        param_3rd = mini
    if param_3rd > maxi:
        param_3rd = maxi
    if BL:
        return random.uniform(mini, maxi), random.uniform(mini, maxi)
    else:
        return param_1st, param_3rd


def getPosePath():
    model_folder = "mhx2"
    label_folder = "newlabel3d"
    pose_folder = "pose"
    model_file = ["Aac01_o", "Aaj01_o", "Aai01_c", "Aah01_o", "Aaf01_o", "Aag01_o", "Aab01_o", "Aaa01_o",
                  "Aad01_o"]  # TrainingSet
    # model_file = [model_folder + "/Aae01_o", model_folder + "/Aaa01_o"] #ImprovementSet1
    # model_file = [model_folder + "/Aae01_o"] #ImprovementSet2
    # model_file = ["Aad01_o"]  # TestSet
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aae01_o"] #TestSet1
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aah01_o"] #TestSet2
    label_file = ["aac01_o", "aaj01_o", "aai01_c", "aah01_o", "aaf01_o", "aag01_o", "aab01_o", "aaa01_o",
                  "aad01_o"]  # TrainingSet
    # label_file = [label_folder + "/aae01_o", label_folder + "/aaa01_o"] #ImprovementSet1
    # label_file = [label_folder + "/aae01_o"] #ImprovementSet2
    # label_file = ["aad01_o"]  # TestSet
    # label_file = [label_folder + "/aad01_o", label_folder + "/aae01_o"] #TestSet1
    # label_file = [label_folder + "/aad01_o", label_folder + "/aah01_o"] #TestSet2
    pose_file = ["P3", "P10", "P9", "P8", "P6", "P7", "P2", "P1", "P4"]  # TrainingSet
    # pose_file = [pose_folder + "/P5", pose_folder + "/P1"] #ImprovementSet1
    # pose_file = [pose_folder + "/P5"] #ImprovementSet2
    # pose_file = ["P4"]  # TestSet
    # pose_file = [pose_folder + "/P4", pose_folder + "/P5"] #TestSet1
    # pose_file = [pose_folder + "/P4", pose_folder + "/P8"] #TestSet2
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
        script = os.path.join(pl.Path(__file__).parent.resolve(), "IEE_V2", "mblab-interface", "scripts",
                              "snt_face_dataset_generation.py")
        data_path = join(pl.Path(__file__).parent.resolve(), "IEE_V2", "mblab_asset_data")
        cmd = [str(blenderPath), "-b", "--log-level", str(0), "-noaudio", "--python", script, "--", str(data_path), "-l", "debug", "-o",
               f"{imgPath}", "--render", "--studio-lights"]  # generate MBLab character
        try:
            devnull = open(join(SimDataPath, str(globalCounter) + "_MBLab_log.txt"), 'w')
            #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            sp.call(cmd, env=os.environ, stdout=devnull, stderr=devnull)
        except Exception as e:
            print("Error in MBLab creating")
            print(e)
            #exit(0)

        filePath = os.path.join(pl.Path(__file__).parent.resolve(), "IEE_V2", "ieekeypoints2.py")
        try:
            devnull = open(join(SimDataPath, str(globalCounter) + "_Blender_log.txt"), 'w')
            #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            cmd = [str(blenderPath), "--background", "-noaudio", "--verbose", str(0), "--python", str(filePath), "--",
                 "--imgPath", str(imgPath)]
            #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            sp.call(cmd, env=os.environ, stdout=devnull, stderr=devnull)
            # str(imgPath)], stdout=subprocess.PIPE)
            shutil.copy(imgPath + ".png", join(SimDataPath, processX(x) + ".png"))
            shutil.copy(imgPath + ".npy", join(SimDataPath, processX(x) + ".npy"))
            shutil.rmtree(join(SimDataPath, str(globalCounter)))
        except Exception as e:
            print("error in Blender scenario creation")
            print(e)
            exit(0)
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
        faceFound = processImage(join(SimDataPath, processX(x)) + ".png", join(pl.Path(__file__).parent.resolve(),
                                                        "IEEPackage", "clsdata", "mmod_human_face_detector.dat"))
    else:
        faceFound = True

    #shutil.copy(join(SimDataPath, processX(x)) + ".png", join(caseFile["GI"], str(caseFile["ID"]), "images", processX(x) + ".png"))
    return join(SimDataPath, processX(x)), faceFound


def generateAnImage(x, caseFile):
    SimDataPath = caseFile["SimDataPath"]
    outPath = caseFile["outputPath"]
    model_folder, label_folder, pose_folder, model_file, label_file, pose_file = getPosePath()
    m = random.randint(0, len(pose_file) - 1)
    m = int(math.floor(x[len(x) - 1]))
    imgPath = join(SimDataPath, processX(x))
    filePath = os.path.join(pl.Path(__file__).parent.resolve(), "IEE_V1", "ieekeypoints.py")
    # filePath = "./ieekeypoints.py"
    t1 = time.time()
    if not isfile(imgPath + ".png"):
        # ls = subprocess.run(['ls', '-a'], capture_output=True, text=True).stdout.strip("\n")
        #print(blenderPath)
        ls = subprocess.run(
            [str(blenderPath), "--background", "-noaudio", "--verbose", str(0), "--python", str(filePath), "--",
             "--path",
             str(join(pl.Path(__file__).parent.resolve(), "IEEPackage")), "--model_folder",
             str(model_folder), "--label_folder", str(label_folder), "--pose_folder", str(pose_folder), "--pose_file",
             str(pose_file[m]), "--label_file", str(label_file[m]), "--model_file", str(model_file[m]), "--imgPath",
             #str(imgPath)])
             str(imgPath)], stdout=subprocess.PIPE)
             #str(imgPath)], capture_output=True, text=True, shell=True).stderr.strip("\n")
        print(ls)
        print(ls.stdout)
        # print(process.stderr)
    # process.wait()
    t2 = time.time()
    # print("Image Generation: ", str(t2-t1)[0:5], end="\r")
    # print(imgPath)
    img = cv2.imread(imgPath + ".png")
    if img is None:
        print("image not found, not processed")
        return imgPath, False
    if len(img) > 128:
        faceFound = processImage(imgPath + ".png", join(pl.Path(__file__).parent.resolve(), "IEEPackage", "clsdata",
                                                        "mmod_human_face_detector.dat"))
    else:
        faceFound = True
    # generator = ieeKP.IEEKPgenerator(model_folder, pose_folder, label_folder)
    # imgPath = generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
    #                                                   cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
    #                                                   label_file[m])
    # print("Image processing", str(time.time()-t2)[0:5])
    return imgPath, faceFound


def doImage(imgPath, caseFile, centroidHM):
    #layersHM, entropy = generateHeatMap(imgPath, caseFile["DNN"], caseFile["datasetName"], caseFile["outputPath"],
    #                                    False, None, None, caseFile["imgExt"], None)
    lable = labelImage(imgPath)
    DNNResult, pred = testModelForImg(caseFile["DNN"], lable, imgPath, caseFile)
    # if imgPath in Dist_Dict:
    #if centroidHM is None:
    #    dist = 0
    #else:
    #    dist = doDistance(centroidHM, layersHM[int(caseFile["selectedLayer"].replace("Layer", ""))], "Euc")
    # Dist_Dict[imgPath] = dist
    #return entropy, DNNResult, pred, lable, dist, layersHM
    return DNNResult, pred, lable
    #return entropy, entropy, entropy, entropy, dist, layersHM

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
    #print(new_img.shape)
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


def processX(x):
    out = str(x[0])[0:5]
    for i in range(1, len(x)):
        out += "_" + str(x[i])[0:5]
    return out



def putMask(imgPath, img, px, py):
    color = (240, 207, 137)
    thick = 23
    img = cv2.line(img, (px[0], py[0]), (px[6], py[6]), color, thick)  # N1 - N2
    img = cv2.line(img, (px[6], py[6]), (px[1], py[1]), color, thick)  # N2 - N3
    img = cv2.line(img, (px[0], py[0]), (px[1], py[1]), color, thick - 4)  # N1 - N3
    img = cv2.line(img, (px[0], py[0]), (px[4], py[4]), color, thick - 4)  # N1 - M1
    img = cv2.line(img, (px[0], py[0]), (px[5], py[5]), color, thick)  # N1 - M3
    img = cv2.line(img, (px[1], py[1]), (px[5], py[5]), color, thick - 4)  # N3 - M3
    img = cv2.line(img, (px[1], py[1]), (px[4], py[4]), color, thick)  # N3 - M1
    img = cv2.line(img, (px[5], py[5]), (px[2], py[2]), color, thick + 2)  # M3 - M2
    img = cv2.line(img, (px[2], py[2]), (px[4], py[4]), color, thick + 2)  # M2 - M1
    img = cv2.line(img, (px[4], py[4]), (px[3], py[3]), color, thick)  # M1 - M4
    img = cv2.line(img, (px[3], py[3]), (px[5], py[5]), color, thick)  # M4 - M3
    img = cv2.line(img, (px[4], py[4]), (px[5], py[5]), color, thick)  # M1 - M3
    label = labelImage(imgPath)
    thick = 4
    length = 35
    angle = 20
    if label == "BottomCenter":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-length-15, py[4]-angle), color, thick)# holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+length-25, py[5]-angle+5), color, thick)# holder (M1 + 15)
        img = img
    elif label == "BottomRight":
        img = cv2.line(img, (px[4], py[4]), (px[4] - length, py[4] - angle), color, thick)  # holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+35, py[5]-20), color, thick)# holder (M1 + 15)
    elif label == "BottomLeft":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-35, py[4]-20), color, thick)# holder (M1 + 15)
        img = cv2.line(img, (px[5], py[5]), (px[5] + length, py[5] - angle), color, thick)  # holder (M1 + 15)
    elif label == "MiddleRight":
        img = cv2.line(img, (px[4], py[4]), (px[4] - length, py[4] - angle), color, thick)  # holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]-35, py[5]-20), color, thick)# holder (M1 + 15)
    elif label == "MiddleLeft":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-35, py[4]-20), color, thick)# holder (M1 + 15)
        img = cv2.line(img, (px[5], py[5]), (px[5] + length, py[5] - angle), color, thick)  # holder (M1 + 15)
    elif label == "MiddleCenter":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-length-15, py[4]-angle-5), color, thick)# holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+length-10, py[5]-angle-5), color, thick)# holder (M1 + 15)
        img = img
    elif label == "TopLeft":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-35, py[4]-20), color, thick)# holder (M1 + 15)
        img = cv2.line(img, (px[5], py[5]), (px[5] + length, py[5] - angle), color, thick)  # holder (M1 + 15)
    elif label == "TopRight":
        img = cv2.line(img, (px[4], py[4]), (px[4] - length, py[4] - angle), color, thick)  # holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+35, py[5]+20), color, thick)# holder (M1 + 15)
    elif label == "TopCenter":
        img = img
        # img = cv2.line(img, (px[4], py[4]), (px[4]-length-10, py[4]-angle-5), color, thick)# holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+length-10, py[5]-angle-5), color, thick)# holder (M1 + 15)
    return img


def putEyeglasses(imgPath, img, px, py):
    color = (240, 207, 137)
    thick = 2
    radius = 12
    if px[0] and py[0] is not None:
        img = cv2.circle(img, (px[0], py[0]), radius, color, thick)
    if px[1] and py[1] is not None:
        img = cv2.circle(img, (px[1], py[1]), radius, color, thick)
    if px[0] and py[0] and px[1] and py[1] is not None:
        img = cv2.line(img, (px[0] + radius, py[0]), (px[1] - radius, py[1]), color, thick)  # N1 - N2
    #img = cv2.line(img, (px[6], py[6]), (px[1], py[1]), color, thick)  # N2 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[1], py[1]), color, thick - 4)  # N1 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[4], py[4]), color, thick - 4)  # N1 - M1
    #img = cv2.line(img, (px[0], py[0]), (px[5], py[5]), color, thick)  # N1 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[5], py[5]), color, thick - 4)  # N3 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[4], py[4]), color, thick)  # N3 - M1
    #img = cv2.line(img, (px[5], py[5]), (px[2], py[2]), color, thick + 2)  # M3 - M2
    #img = cv2.line(img, (px[2], py[2]), (px[4], py[4]), color, thick + 2)  # M2 - M1
    #img = cv2.line(img, (px[4], py[4]), (px[3], py[3]), color, thick)  # M1 - M4
    #img = cv2.line(img, (px[3], py[3]), (px[5], py[5]), color, thick)  # M4 - M3
    #img = cv2.line(img, (px[4], py[4]), (px[5], py[5]), color, thick)  # M1 - M3
    return img

def putSunglasses(imgPath, img, px, py):
    color = (240, 207, 137)
    thick = 17
    radius = 7

    if px[0] and py[0] is not None:
        img = cv2.circle(img, (px[0], py[0]), radius, color, thick)
    if px[1] and py[1] is not None:
        img = cv2.circle(img, (px[1], py[1]), radius, color, thick)
    if px[0] and py[0] and px[1] and py[1] is not None:
        img = cv2.line(img, (px[0] + radius, py[0]), (px[1] - radius, py[1]), color, 2)  # N1 - N2
    #img = cv2.line(img, (px[7], py[7]), (px[8], py[8]), color, thick)  # N1 - N2
    #img = cv2.line(img, (px[6], py[6]), (px[1], py[1]), color, thick)  # N2 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[1], py[1]), color, thick - 4)  # N1 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[4], py[4]), color, thick - 4)  # N1 - M1
    #img = cv2.line(img, (px[0], py[0]), (px[5], py[5]), color, thick)  # N1 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[5], py[5]), color, thick - 4)  # N3 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[4], py[4]), color, thick)  # N3 - M1
    #img = cv2.line(img, (px[5], py[5]), (px[2], py[2]), color, thick + 2)  # M3 - M2
    #img = cv2.line(img, (px[2], py[2]), (px[4], py[4]), color, thick + 2)  # M2 - M1
    #img = cv2.line(img, (px[4], py[4]), (px[3], py[3]), color, thick)  # M1 - M4
    #img = cv2.line(img, (px[3], py[3]), (px[5], py[5]), color, thick)  # M4 - M3
    #img = cv2.line(img, (px[4], py[4]), (px[5], py[5]), color, thick)  # M1 - M3
    return img
