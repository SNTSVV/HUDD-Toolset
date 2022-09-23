#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
#

import sys
#globalDir = sys.path
#localDir = sys.path.insert(1, './')

#sys.path = globalDir
import time
import torch
import os
import shutil
import random
import pandas as pd
from os.path import join, basename, isfile, exists, dirname, isdir
from os import makedirs
from scipy.stats import entropy
from scipy.spatial import distance
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn
import configparser
import statistics as stat
import scipy.stats as sc
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
import argparse
import sys
import math
from sklearn.model_selection import train_test_split
import sklearn.ensemble
#import wittgenstein as lw
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from PIL import Image
from torch.autograd import Variable
from operator import itemgetter
import ntpath
from sklearn import metrics
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist
import cv2
import dlib
import json
import glob
import xlsxwriter
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import torch.optim as optim
import imageio
import torchvision
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
#import dataSupplier as dS
from sklearn.metrics import pairwise_distances
import subprocess
from shutil import rmtree
import hashlib
from PIL import Image
from distutils.dir_util import copy_tree
from os import listdir
class ToTensor(object):
    def __call__(self, img):
        # imagem numpy: C x H x W
        # imagem torch: C X H X W
        #img = img.transpose((0, 1, 2)) all the imgs are processed to 1 channel grayscale
        return torch.from_numpy(img)

class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(PathImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def setupTransformer(dataSetName):
    if dataSetName == "ASL" or dataSetName == "AC":
        data_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),
            transforms.Resize(256),
            # transforms.RandomResizedCrop(256,ratio=(1.0,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif dataSetName == "TS":
        data_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),
            # transforms.Resize(256),
            # transforms.RandomResizedCrop(256,ratio=(1.0,1.0)),
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif dataSetName.startswith("HPD"):
        data_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),
            transforms.Resize(128),
            # transforms.RandomResizedCrop(256,ratio=(1.0,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif dataSetName == "FLD":
        data_transform = transforms.Compose([ToTensor()])
    else:
        data_transform = transforms.Compose([
            transforms.CenterCrop([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    return data_transform


def load_data(data_path, batchSize, maxNum):
    print(data_path)
    data_supplier = dS.Data(data_path, batchSize, True, False, maxNum)
    print(data_supplier)
    print(data_supplier.get_data_iters())
    return data_supplier.get_data_iters()


def getIEEData(f_path, max_num=0, shuffle=True):
    dataset = np.load(f_path, allow_pickle=True)
    dataset = dataset.item()

    x_data = dataset["data"]
    if max_num > 0:
        x_data = x_data[:max_num]
    x_data = x_data.astype(np.float32)
    x_data = x_data / 255.
    # x_data = x_data.reshape((-1,1,x_data.shape[-2], x_data.shape[-1]))
    x_data = x_data[:, np.newaxis]
    # print("x_data shape: ", x_data.shape)

    y_data = dataset["label"]
    if max_num > 0:
        y_data = y_data[:max_num]
    y_data = y_data.astype(np.float32)

    if shuffle:
        r_idx = np.random.permutation(x_data.shape[0])
        x_data = x_data[r_idx]
        y_data = y_data[r_idx]
    x_data = torch.from_numpy(x_data)
    y_data = torch.from_numpy(y_data)
    return x_data, y_data
