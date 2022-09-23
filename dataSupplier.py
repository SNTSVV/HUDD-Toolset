#
# Copyright (c) IEE 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
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



