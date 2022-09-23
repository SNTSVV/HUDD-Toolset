#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#
from HeatmapModule import doDistance, collectHeatmaps, calculate_pixel_distance
from imports import shutil, itemgetter, pd, np, time, torch, os, pdist, shc, metrics, sys, normalize, \
    AgglomerativeClustering, imageio, math, KMeans, join, exists, isfile, cv2, tqdm, Variable, makedirs, basename
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import ieepredict
import dataSupplier as DS
#import hdbscan
from scipy import optimize
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline
from kneed import KneeLocator

def run(caseFile):
    layers = caseFile["layers"]
    outputPath = caseFile["filesPath"]
    mode = caseFile["clustMode"]
    maxCluster = caseFile["maxCluster"]
    start = time.time()
    print("Start Clustering Operation")
    minAvgICD = [0] * len(layers)
    minAvgWICD = [0] * len(layers)
    minAvgS = [0] * len(layers)
    minAvgD = [0] * len(layers)
    clsData = {}
    i = 0
    layersX = list()
    outputPathX = join(outputPath, "ClusterAnalysis_" + str(mode))
    if isfile(join(outputPath, "Layer9HMDistance.xlsx")):
        if not exists(outputPathX):
            os.mkdir(outputPathX)
        for layerX in layers:
            clsPath = join(outputPath, "ClusterAnalysis_" + str(mode), layerX + ".pt")
            if not isfile(clsPath):
                layersX.append(layerX)
                print("Loading heatmaps' distance matrix \n")
                heatMapDistanceExecl = pd.read_excel(join(outputPath, str(layerX) + "HMDistance.xlsx"))
                print("Loaded \n")
                heatMapDistanceExecl.drop(
                    heatMapDistanceExecl.columns[heatMapDistanceExecl.columns.str.contains('unnamed', case=False)],
                    axis=1, inplace=True)
                print("clustering based on Layer " + str(layerX))
                # maxCluster = int(len(heatMapDistanceExecl)/2)
                clsData, minAvgICD[i], minAvgWICD[i] = doClustering(mode, heatMapDistanceExecl, outputPathX, str(layerX)
                                                                    + ".xlsx", maxCluster, layerX, "KG")
                minAvgICD[0] = 1e9
                minAvgWICD[0] = 1e9
                torch.save(clsData, join(outputPathX, str(layerX) + ".pt"))
                sys.stderr.write("Cluster of " + str(layerX) + " saved \n")
                minAvgCluster = pd.DataFrame.from_dict(data=clsData, orient='index')
                writer = pd.ExcelWriter(join(outputPathX, str(layerX) + ".xlsx"), engine='xlsxwriter')
                minAvgCluster.to_excel(writer, sheet_name="ClustersSummary")
                minAvgCluster = pd.DataFrame.from_dict(data=clsData['clusters'], orient='index')
                minAvgCluster.to_excel(writer, sheet_name="ClustersDetails")
                writer.close()
        sys.stderr.write("Cluster of all layers saved \n")
    end = time.time()
    print("Total time consumption of operation Clustering is " + str((end - start) / 60.0) + " minutes.")
    return caseFile




def exportImages_2P(clsData, clsData2, outPathX, caseFile, concepts, layerX):
    imgsPath = []
    for clusterID2 in clsData2['clusters']:
        if len(clsData2['clusters'][clusterID2]['members']) == 1:
            continue
        clusterPath = join(outPathX, str(clusterID2))
            #print("Exists")
        DS.cleanMake(clusterPath, True)
        clusterImages = []
        clusterImages2 = []
        clusterImages2List = []
        for clusterID in clsData2['clusters'][clusterID2]['members']:
            for img in clsData['clusters'][clusterID]['members']:
                if concepts:
                    fileName = img.split(".")[0] + caseFile["imgExt"]
                    srcPath = join(caseFile["DataSetsPath"], str(caseFile["faceSubset"]) + "_Concepts", str(layerX), fileName)
                    dirPath = join(clusterPath, fileName)

                    if caseFile["datasetName"] == "HPD":
                        #origFile = join(caseFile["DataSetsPath"], "TestSet_Backup", str(int(img.split("_")[1])-1) + caseFile["imgExt"])
                        origFile = str(join(caseFile["DataSetsPath"], "TestSet_Backup_M", str((img.split("_")[1])) + caseFile["imgExt"])) #HPD-M
                    else:
                        origFile = join(caseFile["DataSetsPath"], "TestSet_Backup", str(img.split("_")[1]) + caseFile["imgExt"])
                    if origFile not in clusterImages2List:
                        clusterImages2.append(imageio.imread(origFile))
                        clusterImages2List.append(origFile)
                else:
                    fileName = img.split("_")[1] + caseFile["imgExt"]
                    fileClass = img.split("_")[2]
                    fileSource = img.split("_")[0]
                    if fileSource == "Train":
                        srcPath = join(caseFile["trainDataPath"], fileClass, fileName)
                    elif fileSource == "Test":
                        srcPath = join(caseFile["testDataPath"], fileClass, fileName)
                    else:
                        srcPath = join(caseFile["improveDataPath"], fileClass, fileName)
                    dirPath = join(clusterPath, img + caseFile["imgExt"])
                shutil.copy(srcPath, dirPath)
                clusterImages.append(imageio.imread(srcPath))
        imgsPath.append(join(outPathX, str(clusterID2) + '_' + str(len(clusterImages)) + '.gif'))
        imageio.mimsave(join(outPathX, str(clusterID2) + '_' + str(len(clusterImages)) + '.gif'),
                        clusterImages)
        if concepts:
            if (len(clusterImages) / len(clusterImages2List)) > 1.5:
                gifPath = join(outPathX, str(clusterID2) + '_' + str(len(clusterImages2List)) + 'imgsR.gif')
            else:
                gifPath = join(outPathX, str(clusterID2) + '_' + str(len(clusterImages2List)) + 'imgs.gif')
            imgsPath.append(gifPath)
            imageio.mimsave(gifPath, clusterImages2)
    return imgsPath


def getCentroidDists(singleClusters, outputPathX, outputPathY, fileName):
    if not exists(join(outputPathX, fileName)):
        centroidDists = pd.DataFrame()
        for clusterID in singleClusters:
            diffList = []
            centroidHMpath = join(outputPathY, str(clusterID), "centroidHM.pt")
            if not exists(centroidHMpath):
                continue
            hm1 = (torch.load(centroidHMpath)).detach().cpu().numpy()
            for clusterID2 in singleClusters:
                centroidHMpath2 = join(outputPathY, str(clusterID2), "centroidHM.pt")
                if not exists(centroidHMpath2):
                    continue
                hm2 = (torch.load(centroidHMpath2)).detach().cpu().numpy()
                diffList.append(math.sqrt(np.sum(np.power(np.subtract(hm1, hm2), 2))))
            centroidDists[clusterID] = diffList
        if not exists(outputPathX):
            makedirs(outputPathX)
        writer = pd.ExcelWriter(join(outputPathX, fileName), engine='xlsxwriter')
        writer.book.use_zip64()
        centroidDists.to_excel(writer)
        writer.close()
    centroidDists = pd.read_excel(join(outputPathX, fileName))
    centroidDists.drop(centroidDists.columns[centroidDists.columns.str.contains('unnamed', case=False, na=False)],
                       axis=1,
                       inplace=True)
    return centroidDists


def doPass2(outputPathX, layerX, centroidDists):
    if not exists(join(outputPathX, str(layerX) + ".pt")):
        clsData, _, _ = doClustering("AVG", centroidDists, outputPathX, str(layerX) + ".xlsx", 150, layerX, "R")
    else:
        clsData = torch.load(join(outputPathX, str(layerX) + ".pt"))
    singleClusters = []
    for clusterID in clsData['clusters']:
        if len(clsData['clusters'][clusterID]['members']) == 1:
            singleClusters.append(clsData['clusters'][clusterID]['members'][0])
            continue
    return clsData, singleClusters


def saveCentroidHMs(caseFile, concepts, outputPathY, layerX, outPath, clsData):
    for clusterID in clsData['clusters']:
        if len(clsData['clusters'][clusterID]['members']) == 1:
            continue
        clusterImages = []
        clusterImages2 = []
        clusterImages2List = []
        clusterPath = join(outputPathY, str(clusterID))
        if exists(clusterPath):
            continue
        DS.cleanMake(clusterPath, True)
        centroidHM = 0.0
        n = 0
        for img in clsData['clusters'][clusterID]['members']:
            if concepts:
                fileName = img.split(".")[0] + caseFile["imgExt"]
                srcPath = join(caseFile["DataSetsPath"], caseFile["faceSubset"] + "_Concepts", str(layerX), fileName)
                dirPath = join(clusterPath, fileName)
                outDir = join(caseFile["outputPathOriginal"], str(caseFile["faceSubset"]), "ConceptsData")
                outPathX = join(outDir, "ConceptsHM", str(layerX), img.split(".")[0] + ".pt")

                if caseFile["datasetName"] == "HPD":
                    #origFile = join(caseFile["DataSetsPath"], "TestSet_Backup", str(int(img.split("_")[1])-1) + caseFile["imgExt"])
                    origFile = str(join(caseFile["DataSetsPath"], "TestSet_Backup_M", str((img.split("_")[1])) + caseFile["imgExt"])) #HPD-M
                else:
                    origFile = join(caseFile["DataSetsPath"], "TestSet_Backup", str((img.split("_")[1])) + caseFile["imgExt"])
                if origFile not in clusterImages2List:
                    clusterImages2.append(imageio.imread(origFile))
                    clusterImages2List.append(origFile)
            else:
                fileName = img.split("_")[1] + caseFile["imgExt"]
                fileClass = img.split("_")[2]
                fileSource = img.split("_")[0]
                if fileSource == "Train":
                    srcPath = join(caseFile["trainDataPath"], fileClass, fileName)
                elif fileSource == "Test":
                    srcPath = join(caseFile["testDataPath"], fileClass, fileName)
                else:
                    srcPath = join(caseFile["improveDataPath"], fileClass, fileName)
                dirPath = join(clusterPath, img + caseFile["imgExt"])
                outPathX = join(outPath, "Heatmaps", str(layerX), img.split(".")[0] + ".pt")
            shutil.copy(srcPath, dirPath)
            clusterImages.append(imageio.imread(srcPath))
            AN = torch.load(outPathX)
            if n == 0:
                centroidHM = torch.add(AN, 0)
            else:
                centroidHM = torch.add(centroidHM, AN)
            n += 1
        centroidHM = centroidHM / n
        torch.save(centroidHM, join(clusterPath, "centroidHM.pt"))
        imageio.mimsave(join(outputPathY, str(clusterID) + '_' + str(len(clusterImages)) + '.gif'), clusterImages)
        if concepts:
            if (len(clusterImages)/len(clusterImages2List)) > 1.5:
                gifPath = join(outputPathY, str(clusterID) + '_' + str(len(clusterImages2List)) + 'imgsR.gif')
            else:
                gifPath = join(outputPathY, str(clusterID) + '_' + str(len(clusterImages2List)) + 'imgs.gif')
            imageio.mimsave(gifPath, clusterImages2)


def twoPass(caseFile, layerX, concepts, HMDistFile, outPath):
    outputPathY = join(outPath, "2P_FC")
    outputPathZ = join(outPath, "2P_Final")
    if not exists(outputPathZ):
        makedirs(outputPathZ)
    if not exists(join(outputPathY, str(layerX) + ".pt")):
        HMDist1 = pd.read_excel(HMDistFile)
        HMDist1.drop(HMDist1.columns[HMDist1.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        clsData, _, _ = doClustering("WICDWard", HMDist1, outputPathY, str(layerX) + ".xlsx", 150, layerX, "R")
    else:
        clsData = torch.load(join(outputPathY, str(layerX) + ".pt"))
    saveCentroidHMs(caseFile, concepts, outputPathY, layerX, outPath, clsData)
    centroidClusters = []
    for clusterID in clsData['clusters']:
        centroidClusters.append(clusterID)
    singleClusters = centroidClusters
    i = 1
    clsDataList = []
    newImgsPath = []
    while len(singleClusters) > 1:
        outputPathX = join(outPath, "2P_RCC", str(i))
        centroidDists = getCentroidDists(singleClusters, outputPathX, outputPathY, "centroidDists.xlsx")
        clsData2, singleClusters = doPass2(outputPathX, layerX, centroidDists)
        clsDataList.append(clsData2)
        imgsPath = exportImages_2P(clsData, clsData2, outputPathX, caseFile, concepts, layerX)
        print("SingleClusters:", len(singleClusters))
        for img in imgsPath:
            fileName = str(i) + "_" + str(basename(img)).split("_")[0] + "_" + str(basename(img)).split("_")[1]
            shutil.copy(img, join(outputPathZ, fileName))
            newImgsPath.append(join(outputPathZ, fileName))
        i += 1
        break
    return clsData2



def saveIEE_KPs(caseFile, dst, model):

    def forward_hook(self, input, output):
        # print("forward hook..")
        self.X = input[0]
        self.Y = output

    def update(img, x_p, y_p, x_t=0, y_t=0, gt=False):
        height, width = img.shape[0], img.shape[1]
        for idx in [-1, 0, 1]:
            px = max(min(x_p + idx, width - 1), 0)
            if x_t > 0 and y_t > 0:
                tx = max(min(x_t + idx, width - 1), 0)
            for jdx in [-1, 0, 1]:
                py = max(min(y_p + jdx, height - 1), 0)
                if x_t > 0 and y_t > 0:
                    ty = max(min(y_t + jdx, height - 1), 0)
                if width > py > 0 and height > px > 0:
                    if gt: #red
                        img[py, px, 0] = 0
                        img[py, px, 1] = 0
                        img[py, px, 2] = 255
                    else: #blue
                        img[py, px, 0] = 0
                        img[py, px, 1] = 255
                        img[py, px, 2] = 0
                if x_t > 0 and y_t > 0:
                    if width > ty > 0 and height > tx > 0:
                        if gt: #red
                            img[ty, tx, 0] = 0
                            img[ty, tx, 1] = 0
                            img[ty, tx, 2] = 255
                        else: #blue
                            img[ty, tx, 0] = 255
                            img[ty, tx, 1] = 0
                            img[ty, tx, 2] = 0
        return img
    #trainPredict = ieepredict.IEEPredictor(caseFile["trainDataNpy"], caseFile["modelPath"], 0)
    #trainDataSet, mainCounter = trainPredict.load_data(caseFile["trainDataNpy"])
    totalInputs = 0
    testPredict = ieepredict.IEEPredictor(caseFile["testDataNpy"], caseFile["modelPath"], 0)
    testDataSet, _ = testPredict.load_data(caseFile["testDataNpy"])
    mainCounter = 0
    for (inputs, cp_labels) in tqdm(testDataSet):
        totalInputs += len(inputs)
        labels = cp_labels["gm"]
        labels_gt = cp_labels["kps"]
        labels_msk = np.ones(labels_gt.numpy().shape)
        labels_msk[labels_gt.numpy() <= 1e-5] = 0
        if torch.cuda.is_available():
            model = model.cuda()
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        model.conv2d_1.register_forward_hook(forward_hook)
        predictA = model(inputs.float())
        predict_cpu = predictA.cpu()
        predict_cpu = predict_cpu.detach().numpy()
        predict_xy1 = DS.transfer_target(predict_cpu, n_points=DS.n_points)
        predict_xy = np.multiply(predict_xy1, labels_msk)
        inputs_cpu = inputs.cpu()
        inputs_cpu = inputs_cpu.detach().numpy()
        num_sample = inputs_cpu.shape[0]
        for idx in range(num_sample):
            img = inputs_cpu[idx] * 255.
            img = img[0, :]
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            xy = predict_xy[idx]
            lab_xy = labels_gt[idx]
            diff = np.square(np.array(lab_xy) - np.array(xy))
            sum_diff = np.sqrt(diff[:,0] + diff[:,1])
            rightbrow = [2, 3]
            leftbrow = [0, 1]
            mouth = [23, 24, 25, 26]
            righteye = [16, 17, 18, 19, 20, 21, 22]
            lefteye = [9, 10, 11, 12, 13, 14, 15]
            noseridge = [4, 5]
            nose = [6, 7, 8]
            KParray = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
            area = caseFile["faceSubset"]
            if area == "rightbrow":
                KParray = rightbrow
            elif area == "leftbrow":
                KParray = leftbrow
            elif area == "righteye":
                KParray = righteye
            elif area == "lefteye":
                KParray = lefteye
            elif area == "nose":
                KParray = nose
            elif area == "noseridge":
                KParray = noseridge
            elif area == "mouth":
                KParray = mouth
            label = 0
            worst_label = 0
            worst_KP = 0
            for KP in sum_diff:
                if KParray.count(label) > 0:
                    if KP > worst_KP:
                        worst_KP = KP
                        worst_label = label
                label += 1
            for coidx in KParray:
                x_p = int(xy[coidx, 0] + 0.5)
                y_p = int(xy[coidx, 1] + 0.5)
                x_t = int(lab_xy[coidx][0] + 0.5)
                y_t = int(lab_xy[coidx][1] + 0.5)
                if coidx == worst_label:
                    img = update(img, x_p, y_p, x_t, y_t, True)
                #else:
                #    img = update(img, x_p, y_p, x_t, y_t)
                    file_name = join(dst, str(mainCounter)+".png")
            mainCounter += 1
            if not exists(dst):
                os.makedirs(dst)
            if not isfile(file_name):
                cv2.imwrite(file_name, img)

def drawClusters(caseFile, dst, model):
    datasetName = caseFile["datasetName"]
    layers = caseFile["layers"]
    mode = caseFile["clustMode"]
    DataSetsPath = caseFile["DataSetsPath"]
    imgExt = caseFile["imgExt"]
    outputPath = join(caseFile["filesPath"], "ClusterAnalysis_" + str(mode))
    #if datasetName == "FLD":
    #if drawKP:
    #    saveIEE_KPs(caseFile, dst, model)
    #    return
    for layerX in layers:
        if torch.cuda.is_available():
            clsData = torch.load(join(outputPath, str(layerX)) + ".pt")
        else:
            clsData = torch.load(join(outputPath, str(layerX)) + ".pt", map_location = torch.device('cpu'))
        layerPath = join(outputPath, str(layerX))
        if not exists(layerPath):
            os.mkdir(layerPath)
        else:
            shutil.rmtree(layerPath)
            os.mkdir(layerPath)
        allImagesPath = join(layerPath, "AllClusters")
        if not exists(allImagesPath):
            os.mkdir(allImagesPath)
        else:
            shutil.rmtree(allImagesPath)
            os.mkdir(allImagesPath)
        for clusterID in clsData['clusters']:
            clusterImages = []
            clusterCounter = 0
            clusterPath = join(layerPath, str(clusterID))
            if not exists(clusterPath):
                os.mkdir(clusterPath)
            for img in clsData['clusters'][clusterID]['members']:
                clusterCounter += 1
                fileSource = img.split("_")[0]
                if datasetName == "FLD":
                    fileName = img.split("_")[1] + imgExt
                    shutil.copy(join(dst, fileName), join(clusterPath, fileName))
                else:
                    fileName = img.split("_")[1] + imgExt
                    fileClass = img.split("_")[2]
                    if fileSource == "Train":
                        shutil.copy(join(DataSetsPath, "TrainingSet", fileClass, fileName)
                                    , join(clusterPath, fileName))
                    elif fileSource == "Test":
                        shutil.copy(join(DataSetsPath, "TestSet", fileClass, fileName),
                                    join(clusterPath, fileName))
                    else:
                        shutil.copy(join(DataSetsPath, "ImprovementSet", "ImprovementSet", fileClass, fileName),
                                    join(clusterPath, fileName))
                if len(clsData['clusters'][clusterID]['members'])>1:
                    if len(clusterImages) < 140:
                        clusterImages.append(imageio.imread(join(clusterPath, fileName)))
                        clusterImages.append(imageio.imread(join(clusterPath, fileName)))
                        clusterImages.append(imageio.imread(join(clusterPath, fileName)))
            if len(clusterImages)>3:
                imageio.mimsave(join(allImagesPath, 'Cluster' + str(clusterID) + '_' + str(clusterCounter) + '.gif'),
                            clusterImages)
        print("Exported " + str(layerX))

def plotFig(x_axis, y_axis, y_axis2, x_label, y_label, outputPath, layer, pointOne, pointTwo):
    plt.plot(x_axis, y_axis)
    if y_axis2 is not None:
        plt.plot(x_axis, y_axis2)
    if pointOne is not None:
        plt.plot(x_axis[pointOne], y_axis[pointOne], 'ro', ms=5)
    if pointTwo is not None:
        if y_axis2 is not None:
            plt.plot(x_axis[pointTwo], y_axis2[pointTwo], 'go', ms=5)
        else:
            plt.plot(x_axis[pointTwo], y_axis[pointTwo], 'go', ms=5)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(join(outputPath, layer + "_" + y_label + ".png"))
    plt.cla()
    plt.clf()

def KCluster(data, NumClusters):
    print("Kmeans Clustering")
    data_scaled = normalize(data, norm='max')
    print("Distances are normalized")
    kmeans = KMeans(n_clusters=NumClusters, random_state=0).fit(data_scaled)
    label_array = kmeans.labels_
    for i in range(0, len(label_array)):
        label_array[i] = label_array[i] + 1
    return label_array


def HACluster(data, maxCluster, library, linkage, metric, outputPath, layer, selection):
    print("Hirearchial Agglomerative Clustering")
    data_scaled = normalize(data, norm='max')
    print("Distances are normalized")

    #data_scaled = normalize(data)
    if library is None:
        if linkage == "Ward":
            Z = shc.linkage(pdist(data_scaled), metric='euclidean', optimal_ordering=True, method="ward")
        elif linkage == "Avg":
            Z = shc.linkage(pdist(data_scaled), metric='euclidean', optimal_ordering=True, method="average")
        elif linkage == "Complete":
            Z = shc.linkage(pdist(data_scaled), metric='euclidean', optimal_ordering=True, method="complete")
        elif linkage == "Single":
            Z = shc.linkage(pdist(data_scaled), metric='euclidean', optimal_ordering=True, method="single")
        elif linkage == "Centroid":
            Z = shc.linkage(pdist(data_scaled), metric='euclidean', optimal_ordering=True, method="centroid")
        elif linkage == "Median":
            Z = shc.linkage(pdist(data_scaled), metric='euclidean', optimal_ordering=True, method="median")
        elif linkage == "Weighted":
            Z = shc.linkage(pdist(data_scaled), metric='euclidean', optimal_ordering=True, method="weighted")
    print("Linkage", linkage, "data is computed")
    start = 2
    end = maxCluster + 1
    cc_array = [0] * (end - start)
    label_array = [0] * (end - start)
    x_axis = list()
    x_axis2 = list()
    y_axis = list()
    for i in range(start, end):
        print("Clustering:", str(int(100.00*(i/(end-start))))+"%", end="\r")
        labels = shc.fcluster(Z, t=i, criterion='maxclust')
        #labels = (hdbscan.HDBSCAN(min_cluster_size=10)).fit_predict(data_scaled)
        if metric == "Dunn":
            aa = dunn(labels, data_scaled, "farthest", "nearest")
            identifier = 'max'
        if metric == "DunnICD":
            aa = dunnICD(data_scaled, labels)
            identifier = 'max'
        if metric == "ICD":
            aa = getAvgLayer_ICD(data_scaled, labels)
            identifier = 'min'
        if metric == "WICD":
            aa = getAvgLayer_WICD(data_scaled, labels)
            identifier = 'min'
        if metric == "S":
            aa = metrics.silhouette_score(data_scaled, labels, metric='euclidean')
            identifier = 'max'
        if metric == "DBI":
            aa = metrics.davies_bouldin_score(data_scaled, labels)
            identifier = 'min'
        if metric == "AVG":
            aa = getAVG(data_scaled, labels)
            identifier = 'max'
        clustCount = 0
        for x in range(1, i + 1):
            if ((list(labels)).count(x) > 1):
                clustCount += 1
        x_axis2.append(clustCount)
        x_axis.append(i)
        y_axis.append(aa)
        cc_array[i - start] = aa
        label_array[i - start] = labels
    if identifier == 'min':
        c = min(cc_array)
    elif identifier == 'max':
        c = max(cc_array)
    index = [i for i, j in enumerate(cc_array) if j == c]
    index = index[0]
    aa = cc_array[index]
    clusters = index + start
    print("Clustering is done with " + str(metric) + ": " + str(aa) + " and " + str(clusters) + " clusters \n")
    lenX = len(x_axis)
    if lenX % 2 <= 0:
        lenX = lenX - 1
    #smoothY = savgol_filter(y_axis, lenX, 3)
    if len(y_axis) == 1:
        plotFig(x_axis, y_axis, None, "# Clusters", metric, outputPath, layer, None, None)
        #plotFig(x_axis, None, None, "# Clusters", "GradientOfRaw", outputPath, layer, None, None)
        return label_array[0]
    gradient = gradientO4(np.array(y_axis), 4)
    #gradientFit = savgol_filter(gradient, lenX, 3)
    if selection == "KR":
        kn_raw = KneeLocator(x_axis, y_axis, curve='convex', direction='decreasing')
        kneeRaw = kn_raw.knee-start
        plotFig(x_axis, y_axis, None, "# Clusters", metric, outputPath, layer, kneeRaw, None)
        return label_array[kneeRaw]
    elif selection == "KG":
        kn_gradient = KneeLocator(x_axis, gradient, curve='concave', direction='increasing')
        kneeGrad = kn_gradient.knee-start
        plotFig(x_axis, gradient, None, "# Clusters", "GradientOfRaw", outputPath, layer, kneeGrad, None)
        return label_array[kneeGrad]
    elif selection == "R":
        return label_array[index]


def HACluster_SklearnAvg(data: np.array, maxCluster=100):  # data is a obseravation matrix of pair distances
    data_scaled = normalize(data)
    cc_array = [0] * maxCluster
    for i in range(2, maxCluster):
        cluster = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage='average').fit(data_scaled)
        # cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward').fit(data_scaled)
        cc = cluster.fit_predict(data_scaled)
        aa = metrics.silhouette_score(data_scaled, cc, metric='euclidean')
        cc_array[i - 2] = aa
    c = max(cc_array)
    index = [i for i, j in enumerate(cc_array) if j == c]
    index = index[0]
    aa = cc_array[index]
    clusters = index + 2
    cluster = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed', linkage='average').fit(data_scaled)
    # cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward').fit(data_scaled)
    cc = cluster.fit_predict(data_scaled)
    clusterNumber = max(cc) + 1
    aa = metrics.silhouette_score(data_scaled, cc, metric='euclidean')
    print("Clustering is done with silhouette - Sklearn " + str(aa) + " and " + str(clusterNumber) + " clusters \n")
    print(aa)
    for i in range(0, len(cc)):
        cc[i] = cc[i] + 1
    return cc, aa


def HACluster_SklearnSingle(data: np.array, maxCluster=100):  # data is a obseravation matrix of pair distances
    data_scaled = normalize(data)
    cc_array = [0] * maxCluster
    for i in range(2, maxCluster):
        cluster = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage='single').fit(data_scaled)
        # cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward').fit(data_scaled)
        cc = cluster.fit_predict(data_scaled)
        aa = metrics.silhouette_score(data_scaled, cc, metric='euclidean')
        cc_array[i - 2] = aa
    c = max(cc_array)
    index = [i for i, j in enumerate(cc_array) if j == c]
    index = index[0]
    aa = cc_array[index]
    clusters = index + 2
    cluster = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed', linkage='single').fit(data_scaled)
    # cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward').fit(data_scaled)
    cc = cluster.fit_predict(data_scaled)
    clusterNumber = max(cc) + 1
    aa = metrics.silhouette_score(data_scaled, cc, metric='euclidean')
    print("Clustering is done with silhouette - Sklearn " + str(aa) + " and " + str(clusterNumber) + " clusters \n")
    print(aa)
    for i in range(0, len(cc)):
        cc[i] = cc[i] + 1
    return cc, aa


def HACluster_SklearnComplete(data: np.array, maxCluster=100):  # data is a obseravation matrix of pair distances
    data_scaled = normalize(data)
    cc_array = [0] * maxCluster
    for i in range(2, maxCluster):
        cluster = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage='complete').fit(data_scaled)
        # cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward').fit(data_scaled)
        cc = cluster.fit_predict(data_scaled)
        aa = metrics.silhouette_score(data_scaled, cc, metric='euclidean')
        cc_array[i - 2] = aa
    c = max(cc_array)
    index = [i for i, j in enumerate(cc_array) if j == c]
    index = index[0]
    aa = cc_array[index]
    clusters = index + 2
    cluster = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed', linkage='complete').fit(data_scaled)
    # cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward').fit(data_scaled)
    cc = cluster.fit_predict(data_scaled)
    clusterNumber = max(cc) + 1
    aa = metrics.silhouette_score(data_scaled, cc, metric='euclidean')
    print("Clustering is done with silhouette - Sklearn " + str(aa) + " and " + str(clusterNumber) + " clusters \n")
    print(aa)
    for i in range(0, len(cc)):
        cc[i] = cc[i] + 1
    return cc, aa


def getVarCluster(distance: np.array, clusterMember: list, clusterMembersName):
    variance = 0
    for m1 in range(0, len(clusterMember)):
        for m2 in range(m1 + 1, len(clusterMember)):
            indexM1 = clusterMembersName.get_loc(clusterMember[m1])
            indexM2 = clusterMembersName.get_loc(clusterMember[m2])
            variance += distance[indexM1][indexM2] ** 2
    return variance


def getSumDistanceCluster(distance: np.array, clusterMember: list, clusterMembersName):
    sumDistance = 0
    for m1 in range(0, len(clusterMember)):
        for m2 in range(m1 + 1, len(clusterMember)):
            indexM1 = clusterMembersName.get_loc(clusterMember[m1])
            indexM2 = clusterMembersName.get_loc(clusterMember[m2])
            sumDistance += distance[indexM1][indexM2]
    return sumDistance


def getVarClusterWithAddition(clusterMember: list, currentVar: int, newMember: str, heatMapTestSet,
                              heatmapTraningSet, metric):
    variance = currentVar
    for m1 in range(0, len(clusterMember)):
        variance += doDistance(heatMapTestSet[m1], heatmapTraningSet[newMember], metric)
    return variance

def dunnICD(distance, cc):
    #return getAvgCluster_ICD(distance, cc)/getAvgLayer_ICD(distance, cc)
    return getAvgCluster_ICD(distance, cc)/getFarthestLayer_ICD(distance, cc)
    #return getAvgCluster_ICD(distance, cc)/getAvgLayer_WICD(distance, cc)

def getAvgCluster_ICD(distance: np.array, cc):
    ClosestPairDists = list()
    for label1 in range(1, max(cc)+ 1):
        groupLen1 = list(cc).count(label1)
        PairDists = list()
        if groupLen1 > 1:
            groupIndex1 = [0] * groupLen1
            j = 0
            for i in range(0, len(cc)):
                if (cc[i] == label1):
                    groupIndex1[j] = i
                    j = j + 1
            for label2 in range(1, max(cc)+1):
                if label1 != label2:
                    groupLen2 = list(cc).count(label2)
                    if groupLen2 > 1:
                        groupIndex2 = [0] * groupLen2
                        j = 0
                        for i in range(0, len(cc)):
                            if (cc[i] == label2):
                                groupIndex2[j] = i
                                j = j + 1
                        for index in groupIndex1:
                            for index2 in groupIndex2:
                                PairDists.append(distance[index][index2])
            if not len(PairDists) == 0:
                ClosestPairDists.append(min(PairDists))
    return sum(ClosestPairDists)/len(ClosestPairDists)

def getFarthestLayer_ICD(distance: np.array, cc):
    Clusters_ICD = list()
    for label in range(1, max(cc) + 1):
        groupLen = list(cc).count(label)
        # weight = groupLen / len(cc)
        numPairs = int((groupLen * (groupLen - 1)) / 2)
        Dists = [0] * numPairs
        groupIndex = [0] * groupLen
        j = 0

        for i in range(0, len(cc)):
            if (cc[i] == label):
                groupIndex[j] = i
                j = j + 1
        k = 0
        for index in groupIndex:
            for index2 in groupIndex:
                if (index2 > index):
                    Dists[k] = distance[index][index2]
                    k = k + 1
        if (groupLen > 1):
            Clusters_ICD.append(max(Dists))

    return sum(Clusters_ICD)/len(Clusters_ICD)

def getAvgLayer_ICD(distance: np.array, cc):
    Clusters_ICD = list()
    for label in range(1, max(cc) + 1):
        groupLen = list(cc).count(label)
        # weight = groupLen / len(cc)
        numPairs = int((groupLen * (groupLen - 1)) / 2)
        Dists = [0] * numPairs
        groupIndex = [0] * groupLen
        j = 0

        for i in range(0, len(cc)):
            if (cc[i] == label):
                groupIndex[j] = i
                j = j + 1
        k = 0
        for index in groupIndex:
            for index2 in groupIndex:
                if (index2 > index):
                    Dists[k] = distance[index][index2]
                    k = k + 1
        if (groupLen > 1):
            Clusters_ICD.append(sum(Dists)/numPairs)
    return sum(Clusters_ICD)/len(Clusters_ICD)

def getAVG(distance: np.array, cc):
    Clusters_ICD = list()
    Clusters_Indices = list()
    for label in range(1, max(cc) + 1):
        groupIndex = list()
        for i in range(0, len(cc)):
            if (cc[i] == label):
                groupIndex.append(i)
        Clusters_Indices.append(groupIndex)

    k = 0
    for cluster_indices1 in Clusters_Indices:
        j = 0
        for cluster_indices2 in Clusters_Indices:
            if (j > k):
                dist = list()
                for index1 in cluster_indices1:
                    for index2 in cluster_indices2:
                        dist.append(distance[index1][index2])
                Clusters_ICD.append(min(dist))
            j += 1
        k += 1

    return sum(Clusters_ICD)/len(Clusters_ICD)

def getAvgLayer_NewICD(distance: np.array, cc):
    n_clusters = 0
    ICDlayer = 0
    for label in range(1, max(cc) + 1):
        groupLen = list(cc).count(label)
        # weight = groupLen / len(cc)
        numPairs = int((groupLen * (groupLen - 1)) / 2)
        Dists = [0] * numPairs
        groupIndex = [0] * groupLen
        j = 0

        for i in range(0, len(cc)):
            if (cc[i] == label):
                groupIndex[j] = i
                j = j + 1
        k = 0
        for index in groupIndex:
            for index2 in groupIndex:
                if (index2 > index):
                    Dists[k] = distance[index][index2]
                    k = k + 1
        if (groupLen > 1):
            Avglabel = sum(Dists) / numPairs
            ICDc = Avglabel * groupLen
            # ICDc = Avglabel * weight
            ICDlayer = ICDlayer + ICDc
            n_clusters += 1
    return ICDlayer / n_clusters


def getAvgLayer_WICD(distance: np.array, cc):
    n_clusters = max(cc)
    ICDlayer = 0
    for label in range(1, n_clusters + 1):
        groupLen = list(cc).count(label)
        weight = groupLen / len(cc)
        numPairs = int((groupLen * (groupLen - 1)) / 2)
        Dists = [0] * numPairs
        groupIndex = [0] * groupLen
        j = 0

        for i in range(0, len(cc)):
            if (cc[i] == label):
                groupIndex[j] = i
                j = j + 1
        k = 0
        for index in groupIndex:
            for index2 in groupIndex:
                if (index2 > index):
                    Dists[k] = distance[index][index2]
                    k = k + 1
        if (groupLen > 1):
            Avglabel = sum(Dists) / numPairs
            # ICDc = Avglabel
            ICDc = Avglabel * weight
            ICDlayer = ICDlayer + ICDc
    return ICDlayer / n_clusters



def doClustering(mode, heatmapsDistance: dict, outPutPath: str, outputFile: str, maxCluster: int, layer, selection):
    if not exists(outPutPath):
        makedirs(outPutPath)
    sys.stderr.write("Clustering ... \n")
    clusters = {}
    clusters['selected_ICD'] = False
    clusters['selected_WICD'] = False
    clusters['selected_S'] = False
    clusters['selected_Dunn'] = False
    if mode.startswith('AVG'):
        metric = 'AVG'
        linkage = 'Ward'
    if mode.startswith('ICD'):
        metric = 'ICD'
        if mode == 'ICDWard':
            linkage = 'Ward'
        if mode == 'ICDAvg':
            linkage = 'Avg'
        if mode == 'ICDSingle':
            linkage = 'Single'
    if mode.startswith('WICD'):
        metric = 'WICD'
        if mode == 'WICDWard':
            linkage = 'Ward'
        if mode == 'WICDAvg':
            linkage = 'Avg'
    if mode.startswith('Dunn'):
        metric = 'Dunn'
        if mode == 'DunnWard':
            linkage = 'Ward'
        if mode == 'DunnAvg':
            linkage = 'Avg'
    if mode.startswith('DunnICD'):
        metric = 'DunnICD'
        if mode == 'DunnICDWard':
            linkage = 'Ward'
        if mode == 'DunnICDAvg':
            linkage = 'Avg'
    if mode.startswith('DBI'):
        metric = 'DBI'
        if mode == 'DBIWard':
            linkage = 'Ward'
        if mode == 'DBIAvg':
            linkage = 'Avg'
    if mode.startswith('S'):
        metric = 'S'
        if mode == 'SWard':
            linkage = 'Ward'
        if mode == 'SAvg':
            linkage = 'Avg'
    if mode.startswith('K'):
        metric = 'K'
        linkage = None
    data = heatmapsDistance.values
    print(metric)
    print(linkage)
    if metric == 'K':
        cc = KCluster(data, maxCluster)
    else:
        cc = HACluster(data, maxCluster, None, linkage, metric, outPutPath, layer, selection)
    data_scaled = normalize(data, norm='max')
    numAC = 0
    for label in range(1, max(cc) + 1):
        if list(cc).count(label) > 1:
            numAC += 1
    if numAC == 0:
        WICD = 1e9
        ICD = 1e9
        S = -1e9
        DunnIndex = -1e9
        DBI = 1e9
    else:
        WICD = getAvgLayer_WICD(data, cc)
        ICD = getAvgLayer_ICD(data, cc)
        S = metrics.silhouette_score(data_scaled, cc, metric='euclidean')
        DunnIndex = dunn(cc, data_scaled, "farthest", "nearest")
        DBI = metrics.davies_bouldin_score(data_scaled, cc)
    clusters['WeightedavgLayer'] = WICD
    clusters['avgLayer'] = ICD
    # clusters['avgLayer'] = getAvgLayer_NewICD(heatmapsDistance.values, cc)
    clusters['silhouette'] = S
    clusters['dunn'] = DunnIndex
    clusters['DBI'] = DBI
    clusters['label list'] = cc
    index = 0
    for clusterID in cc:
        if not 'clusters' in clusters:
            clusters['clusters'] = {}
        if not clusterID in clusters['clusters']:
            clusters['clusters'][clusterID] = {}
        if not 'members' in clusters['clusters'][clusterID]:
            clusters['clusters'][clusterID]['members'] = []
        clusters['clusters'][clusterID]['members'].append(heatmapsDistance.columns[index])
        index = index + 1

    avgDistLayers = []
    nonSingleClusters = 0
    for clusterID in clusters['clusters']:
        clusters['clusters'][clusterID]['variance'] = getVarCluster(heatmapsDistance.values,
                                                                    clusters['clusters'][clusterID][
                                                                        'members'],
                                                                    heatmapsDistance.columns)
        clusters['clusters'][clusterID]['length'] = len(clusters['clusters'][clusterID]['members'])
        if len(clusters['clusters'][clusterID]['members']) > 1:
            nonSingleClusters += 1
        clusters['clusters'][clusterID]['sumDistance'] = getSumDistanceCluster(
            heatmapsDistance.values, clusters['clusters'][clusterID]['members'],
            heatmapsDistance.columns)
        clusters['clusters'][clusterID]['errorSum'] = clusters['clusters'][clusterID]['variance'] / \
                                                      clusters['clusters'][clusterID]['length']
        avgDistLayers.append(
            clusters['clusters'][clusterID]['sumDistance'] / clusters['clusters'][clusterID][
                'length'])
    print("Number of non-single clusters:", nonSingleClusters)
    minAvgCluster = pd.DataFrame.from_dict(data=clusters, orient='index')
    writer = pd.ExcelWriter(outPutPath + "/" + outputFile, engine='xlsxwriter')
    minAvgCluster.to_excel(writer, sheet_name="ClustersSummary")

    minAvgCluster = pd.DataFrame.from_dict(data=clusters['clusters'], orient='index')
    minAvgCluster.to_excel(writer, sheet_name="ClustersDetails")

    writer.close()
    torch.save(clusters, outPutPath + "/" + outputFile.split(".")[0] + ".pt")
    return clusters, ICD, WICD


def inter_cluster_distances(labels, distances, method='nearest'):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                    (not farthest and
                     distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                    (farthest and
                     distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculates cluster diameters
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """
    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, distances, diameter_method='farthest',
         cdist_method='nearest'):
    """
    Dunn index for cluster validation (larger is better).

    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace

    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)` is the diameter of cluster :math:`c_k`.
    Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their closest elements. Cluster diameter can be defined as the mean distance between all elements in the cluster, between all elements to the cluster centroid, or as the distance between the two furthest elements.
    The higher the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart (large :math:`d \\left( c_i,c_j \\right)`).
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param diameter_method: see :py:function:`diameter` `method` parameter
    :param cdist_method: see :py:function:`diameter` `method` parameter

    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    #print(ic_distances)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter


def postCluster(heatmapPath, selectedLayer, selectedLayerClusters, heatmapsDistance, metric):
    testHM, imgListX = collectHeatmaps(heatmapPath, selectedLayer)

    for i in range(0, 35):
        clsDict, selectedLayerClusters = getClusterData(selectedLayerClusters, testHM, metric)
        bestfit = 0
        bestfitcls1 = None
        bestfitcls2 = None
        for cls1 in clsDict:
            for cls2 in clsDict[cls1]:
                if cls2 is not None:
                    if clsDict[cls1][cls2] >= bestfit:
                        bestfit = clsDict[cls1][cls2]
                        bestfitcls1 = cls1
                        bestfitcls2 = cls2
        print(bestfitcls1, bestfitcls2, bestfit)

        newSum = selectedLayerClusters['clusters'][bestfitcls1]['sumDistance'] \
                 + selectedLayerClusters['clusters'][bestfitcls2]['sumDistance']
        for img1 in selectedLayerClusters['clusters'][bestfitcls1]['members']:
            for img2 in selectedLayerClusters['clusters'][bestfitcls2]['members']:
                if img1 != img2:
                    newSum += doDistance(testHM[img1], testHM[img2], metric)
        selectedLayerClusters['clusters'][bestfitcls1]['sumDistance'] = newSum
        for img in selectedLayerClusters['clusters'][bestfitcls2]['members']:
            selectedLayerClusters['clusters'][bestfitcls1]['members'].append(img)
        del selectedLayerClusters['clusters'][bestfitcls2]
        counter = 0
        for clusterID in selectedLayerClusters['clusters']:
            counter = counter + 1
        print(counter)


def getClusterData(selectedLayerClusters, testHM, metric):
    clsDict = {}
    for clusterID in selectedLayerClusters['clusters']:
        clsDict[clusterID] = {}
        deltaICD = {}
        Sum1 = selectedLayerClusters['clusters'][clusterID]['sumDistance']
        Len1 = len(selectedLayerClusters['clusters'][clusterID]['members'])
        Pairs1 = ((Len1 * (Len1 - 1)) / 2)
        if Len1 > 1:
            oldICD = Sum1 / Pairs1
            bestClst = None
            bestICD = oldICD
            for clusterID2 in selectedLayerClusters['clusters']:
                # if clusterID != clusterID2:
                Sum2 = selectedLayerClusters['clusters'][clusterID2]['sumDistance']
                Len2 = len(selectedLayerClusters['clusters'][clusterID2]['members'])
                Pairs2 = ((Len2 * (Len2 - 1)) / 2)
                newSum = Sum1 + Sum2
                newPairs = Pairs1 + Pairs2
                for img1 in selectedLayerClusters['clusters'][clusterID]['members']:
                    for img2 in selectedLayerClusters['clusters'][clusterID2]['members']:
                        if img1 != img2:
                            newSum += doDistance(testHM[img1], testHM[img2], metric)
                            newPairs += 1
                newICD = newSum / newPairs
                deltaICD[clusterID2] = oldICD - newICD
                # if newICD <= oldICD:
                # if newICD <= bestICD:
                # print(clusterID, clusterID2, oldICD, newICD)
                # bestClst = clusterID2
                # bestICD = newICD
                # clsDict[clusterID][clusterID2] = newICD
                # else:
                # clsDict[clusterID][clusterID2] = 0
            bestICD = 0
            for clusterID2 in deltaICD:
                if deltaICD[clusterID2] >= bestICD:
                    # print(clusterID, clusterID2, deltaICD[clusterID2])
                    bestICD = deltaICD[clusterID2]
                    bestClst = clusterID2
            clsDict[clusterID][bestClst] = bestICD
    return clsDict, selectedLayerClusters


# 4th order accurate gradient function based on 2nd order version from http://projects.scipy.org/scipy/numpy/browser/trunk/numpy/lib/function_base.py
def gradientO4(f, *varargs):
    """Calculate the fourth-order-accurate gradient of an N-dimensional scalar function.
    Uses central differences on the interior and first differences on boundaries
    to give the same shape.
    Inputs:
      f -- An N-dimensional array giving samples of a scalar function
      varargs -- 0, 1, or N scalars giving the sample distances in each direction
    Outputs:
      N arrays of the same shape as f giving the derivative of f with respect
       to each dimension.
    """
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n == 0:
        dx = [1.0] * N
    elif n == 1:
        dx = [varargs[0]] * N
    elif n == N:
        dx = list(varargs)
    else:
        raise(SyntaxError, "invalid number of arguments")

    # use central differences on interior and first differences on endpoints

    # print dx
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)] * N
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = np.zeros(f.shape, f.dtype.char)

        slice0[axis] = slice(2, -2)
        slice1[axis] = slice(None, -4)
        slice2[axis] = slice(1, -3)
        slice3[axis] = slice(3, -1)
        slice4[axis] = slice(4, None)
        # 1D equivalent -- out[2:-2] = (f[:4] - 8*f[1:-3] + 8*f[3:-1] - f[4:])/12.0
        out[slice0] = (f[slice1] - 8.0 * f[slice2] + 8.0 * f[slice3] - f[slice4]) / 12.0

        slice0[axis] = slice(None, 2)
        slice1[axis] = slice(1, 3)
        slice2[axis] = slice(None, 2)
        # 1D equivalent -- out[0:2] = (f[1:3] - f[0:2])
        out[slice0] = (f[slice1] - f[slice2])

        slice0[axis] = slice(-2, None)
        slice1[axis] = slice(-2, None)
        slice2[axis] = slice(-3, -1)
        ## 1D equivalent -- out[-2:] = (f[-2:] - f[-3:-1])
        out[slice0] = (f[slice1] - f[slice2])

        # divide by step size
        outvals.append(out / dx[axis])

        # reset the slice object in this dimension to ":"
        slice0[axis] = slice(None)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if N == 1:
        return outvals[0]
    else:
        return outvals


def laplacian_O3(f, *varargs):
    """ Third order accurate, 5-point formula. Not really laplacian, but second derivative along each axis. Sum the outputs to get laplacian."""
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n == 0:
        dx = [1.0] * N
    elif n == 1:
        dx = [varargs[0]] * N
    elif n == N:
        dx = list(varargs)
    else:
        raise(SyntaxError, "invalid number of arguments")

    # use central differences on interior and first differences on endpoints

    # print dx
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)] * N
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = np.zeros(f.shape, f.dtype.char)

        # http://www.sitmo.com/eq/262 (3rd order accurate)
        slice0[axis] = slice(2, -2)
        slice1[axis] = slice(None, -4)
        slice2[axis] = slice(1, -3)
        slice3[axis] = slice(3, -1)
        slice4[axis] = slice(4, None)
        # 1D equivalent -- out[2:-2] = (-f[:-4] + 16*f[1:-3] + -30*f[2:-2] + 16*f[3:-1] - f[4:])/12.0
        out[slice0] = (-f[slice1] + 16.0 * f[slice2] - 30.0 * f[slice0] + 16.0 * f[slice3] - f[slice4]) / 12.0

        # http://www.sitmo.com/eq/260 (2nd order accurate; there's also a 3rd order accurate that requires 5 points)
        slice0[axis] = slice(None, 2)
        slice1[axis] = slice(3, 5)
        slice2[axis] = slice(2, 4)
        slice3[axis] = slice(1, 3)
        # 1D equivalent -- out[0:2] = 2*f[0:2] - 5*f[1:3] + 4*f[2:4] - f[3:5]
        out[slice0] = (2.0 * f[slice0] - 5.0 * f[slice3] + 4.0 * f[slice2] - f[slice1])

        slice0[axis] = slice(-2, None)
        slice1[axis] = slice(-5, -3)
        slice2[axis] = slice(-4, -2)
        slice3[axis] = slice(-3, -1)
        # 1D equivalent -- out[0:2] = 2*f[0:2] - 5*f[1:3] + 4*f[2:4] - f[3:5]
        out[slice0] = (2.0 * f[slice0] - 5.0 * f[slice3] + 4.0 * f[slice2] - f[slice1])

        # divide by step size
        axis_dx = dx[axis]
        outvals.append(out / (axis_dx * axis_dx))

        # reset the slice object in this dimension to ":"
        slice0[axis] = slice(None)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if N == 1:
        return outvals[0]
    else:
        return outvals


def test_laplacian_03():
    """ Simple sanity check on whether the centers and edges have signs correct with the right value"""

    increasing = np.arange(10) - 5
    decreasing = -increasing
    ones = np.ones((10,))
    x_grid = ones[:9] * increasing[:, None]
    y_grid = ones[:, None] * increasing[None, :9]
    grid = x_grid + y_grid
    d2dx2, d2dy2 = laplacian_O3(grid ** 3.0, 1, 1)
    assert (d2dx2 == 6.0 * grid).all()
    assert (d2dy2 == 6.0 * grid).all()

    d2dx2, d2dy2 = laplacian_O3(grid ** 2.0, 3.0, 5.0)
    assert (d2dx2 == 2.0 / 3.0 / 3.0).all()
    assert (d2dy2 == 2.0 / 5.0 / 5.0).all()

    print("Laplacian checks ok")