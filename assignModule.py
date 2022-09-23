#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#
import shutil

import imageio

import HeatmapModule
import testModule
import dataSupplier as DS
from imports import sys, Variable, pd, torch, time, os, itemgetter, random, np, setupTransformer, math, join, exists, \
    basename, dirname, isfile


def run(caseFile_):
    global caseFile, centroidHMs, centroidRadius, clusterIDX
    caseFile = caseFile_
    start = time.time()
    outputPath = caseFile["filesPath"]
    dnn = caseFile["DNN"]
    dnn.eval()
    datasetName = caseFile["datasetName"]
    selectedLayer = caseFile["selectedLayer"]
    assignMode = caseFile["assignMode"]
    area = caseFile["faceSubset"]
    imgExt = caseFile["imgExt"]
    improveNpy = caseFile["improveDataNpy"]
    FLD = caseFile["FLD"]
    #print("FLD", FLD)
    TPtotal = 0
    FPtotal = 0
    FNtotal = 0
    TestCounter = 0
    TrainClusters = 0
    TestClusters = 0
    TestTrainClusters = 0
    retrainHMPath = join(caseFile["outputPath"], "trainHeatmaps", selectedLayer)
    assignedPath = join(caseFile["outputPath"], "T", "UnsafeSet")
    makeFolder(retrainHMPath)
    RemainingTime = "N/A"
    numClusters = 0
    if not isfile(caseFile["assignPTFile"]):
        print("Loading HM distance file for the selected layer.")
        heatMapDistanceExecl = pd.read_excel(join(outputPath, str(selectedLayer) + "HMDistance.xlsx"))
        heatMapDistanceExecl.drop(
            heatMapDistanceExecl.columns[heatMapDistanceExecl.columns.str.contains('unnamed', case=False)],
            axis=1, inplace=True)
        caseFile["assImages"] = []
        caseFile["notAssImages"] = []
        caseFile["actualCluster"] = []
        caseFile["expctedCluster"] = []
        TestSetCheck = False
        x = 0
        totalCounter = 0
        getClusterData(caseFile, heatMapDistanceExecl)
        clsWithAssImages = torch.load(caseFile["clsPath"])
        print("numClusters:", len(clsWithAssImages['clusters']))
        loadBar = 0.0
        start = time.time()
        TestClusters = 0
        retrainLength = len(caseFile["retrainList"])
        for trainImage in caseFile["retrainList"]:
            errImage, fileName, retrainImage = nameMapper(trainImage)
            candidateClusterID = -1
            for clusterID in clsWithAssImages['clusters']:
                if 'selected' not in clsWithAssImages['clusters'][clusterID]:
                    clsWithAssImages['clusters'][clusterID]['selected'] = []
                if 'assigned' not in clsWithAssImages['clusters'][clusterID]:
                    clsWithAssImages['clusters'][clusterID]['assigned'] = []
                for testImage in clsWithAssImages['clusters'][clusterID]['members']:
                    if testImage == fileName:
                        candidateClusterID = clusterID
            caseFile["expctedCluster"].append(candidateClusterID)
        for trainImage in caseFile["retrainList"]:
            totalCounter += 1
            if x / int(retrainLength * 0.01) == 1:
                end = time.time()
                RemainingTime = str(math.ceil(((100.0*(end - start) / 60.0) * (100 - loadBar)))) + " mins."
                loadBar += 1.0
            layerIndex = caseFile["layerIndex"]
            errImage, fileName, retrainImage = nameMapper(trainImage)
            heatMap = HeatmapModule.safeHM(join(retrainHMPath, errImage), layerIndex, trainImage,
                                           dnn, datasetName, "", False, area, improveNpy, imgExt, FLD)
            clsWithAssImages, breakFlag = assign(trainImage, heatMap)
            if breakFlag:
                break
            else:
                print("Checked:", str(loadBar) + "%", "Assigned:", str(len(caseFile["assImages"])) + " ETA: <" +
                      str(RemainingTime), end="\r")

        totalError = 0
        TrainClusters = 0
        TestTrainClusters = 0
        TestTrainAssigned = 0
        TestAssigned = 0
        TrainAssigned = 0
        clustCounter = 0
        clustersDetails = {}
        for clusterID in clusterIDX:
            TestCounter = 0
            TrainCounter = 0
            clustersDetails[clusterID] = {}
            trainflag = False
            testflag = False
            for member in clsWithAssImages['clusters'][clusterID]['members']:
                totalError += 1
                if member.startswith("Test_"):
                    testflag = True
                    TestCounter += 1
                else:
                    trainflag = True
                    TrainCounter += 1
            clustCounter += 1
            if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                clustersDetails[clusterID]['#assigned'] = len(clsWithAssImages['clusters'][clusterID]['assigned'])
            else:
                clustersDetails[clusterID]['#assigned'] = 0

            if clustersDetails[clusterID]['#assigned'] > 0:
                clustersDetails[clusterID]["ass/notass"] = True
            else:
                clustersDetails[clusterID]["ass/notass"] = False
            if testflag and trainflag:
                clustersDetails[clusterID]['type'] = "TestTrain"
                clustersDetails[clusterID]['#test'] = TestCounter
                clustersDetails[clusterID]['#train'] = TrainCounter
                if clustersDetails[clusterID]["ass/notass"]:
                    TestTrainAssigned += 1
                TestTrainClusters += 1
            elif testflag:
                clustersDetails[clusterID]['type'] = "Test"
                clustersDetails[clusterID]['#test'] = TestCounter
                clustersDetails[clusterID]['#train'] = 0
                if clustersDetails[clusterID]["ass/notass"]:
                    TestAssigned += 1
                TestClusters += 1
            elif trainflag:
                clustersDetails[clusterID]['type'] = "Train"
                clustersDetails[clusterID]['#test'] = 0
                clustersDetails[clusterID]['#train'] = TrainCounter
                if clustersDetails[clusterID]["ass/notass"]:
                    TrainAssigned += 1
                TrainClusters += 1
            print("Clust:" + str(clusterID), "type: " + str(clustersDetails[clusterID]['type']),
                  "#test: " + str(clustersDetails[clusterID]['#test']),
                  "#train: " + str(clustersDetails[clusterID]['#train']),
                  "#assigned: " + str(clustersDetails[clusterID]['#assigned']))
        print("Total Clusters", clustCounter)
        assignedImageCluster1 = pd.DataFrame.from_dict(data=clsWithAssImages['clusters'], orient='index')
        assignedImageCluster3 = pd.DataFrame.from_dict(data=clsWithAssImages, orient='index')
        makeFolder(basename(caseFile["assignXLFile"]))
        writer = pd.ExcelWriter(caseFile["assignXLFile"], engine='xlsxwriter')
        assignedImageCluster1.to_excel(writer, sheet_name="Assignment Result Summary")
        assignedImageCluster3.to_excel(writer, sheet_name="Assignment Result Clusters")
        writer.close()
        torch.save(clsWithAssImages, join(caseFile["assignPTFile"]))
        clsters = list()
        clusterDistrib = list()
        for clusterID in clsWithAssImages['clusters']:
            clusterDistrib.append(clsters)
            if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                clusterDistrib.append(clustLen)
            else:
                clusterDistrib.append(0)
        print("Clusters Distribution", clusterDistrib)
        print("Total Clusters", len(clsWithAssImages['clusters']))
        clusterList = list()
        for clusterID in clsWithAssImages['clusters']:
            clusterList.append(clusterID)
        clusterList.append(-1)
        TPtotal = 0
        FPtotal = 0
        FNtotal = 0
        if TestSetCheck:
            for clusterID in clsWithAssImages['clusters']:
                TP = 0
                FP = 0
                FN = 0
                x = 0
                for aCluster in caseFile["actualCluster"]:
                    eCluster = caseFile["expctedCluster"][x]
                    x += 1
                    if eCluster == clusterID:
                        if aCluster == clusterID:
                            TP += 1
                            TPtotal += 1
                        else:
                            FN += 1
                            FNtotal += 1
                    else:
                        if aCluster == clusterID:
                            FP += 1
                            FPtotal += 1
            print(TPtotal, FPtotal, FNtotal)
            print("TrainClusters:", TrainClusters, TrainClusters / len(clsWithAssImages['clusters']))
            print("TestClusters:", TestClusters, TestClusters / len(clsWithAssImages['clusters']))
            print("TestTrainClusters:", TestTrainClusters, TestTrainClusters / len(clsWithAssImages['clusters']))

    clsWithAssImages = torch.load(caseFile["assignPTFile"])
    for clusterID in clsWithAssImages['clusters']:
        clusterImages = []
        for img in clsWithAssImages['clusters'][clusterID]['assigned']:
            if not exists(join(assignedPath, str(clusterID))):
                os.makedirs(join(assignedPath, str(clusterID)))
            shutil.copy(img, join(assignedPath, str(clusterID), basename(img)))
            clusterImages.append(imageio.imread(img))
        imageio.mimsave(join(assignedPath, str(clusterID) + '_' + str(len(clusterImages)) + '.gif'), clusterImages)
    caseFile[assignMode] = {}
    caseFile[assignMode]["assImages"] = []
    #for clusterID in clsWithAssImages['clusters']:
    #    for img in clsWithAssImages['clusters'][clusterID]['assigned']:
    #        caseFile[assignMode]["assImages"].append(img)
    if len(caseFile["retrainList"]) == 0:
        caseFile[assignMode]["%Assigned"] = 0.0
    else:
        caseFile[assignMode]["%Assigned"] = 100.00 * (len(caseFile[assignMode]["assImages"]) / len(caseFile["retrainList"]))
    #print(caseFile[assignMode]["%Assigned"])
    caseFile[assignMode]["TP"] = TPtotal
    caseFile[assignMode]["FP"] = FPtotal
    caseFile[assignMode]["FN"] = FNtotal
    caseFile[assignMode]["totalErrImgs"] = TestCounter
    caseFile[assignMode]["trainClusters"] = TrainClusters
    caseFile[assignMode]["testClusters"] = TestClusters
    caseFile[assignMode]["ttClusters"] = TestTrainClusters
    caseFile[assignMode]["numClusters"] = numClusters
    #print("INFO-assImages-caseFile:", len(caseFile[assignMode]["assImages"]))
    torch.save(caseFile, caseFile["caseFile"])
    end = time.time()
    #print("Assigned " + str(caseFile[assignMode]["%Assigned"]) + " % of ImprovementSet")
    #print("Assigning images into clutsres is finished \n")
    #print("Total time consumption of operation Assigning Images is " + str((end - start) / 60.0) + " minutes.")
    return caseFile


def nameMapper(trainImage):
    global caseFile
    retrainImage = basename(dirname(trainImage)) + "_" + basename(trainImage).split(".")[0]
    if caseFile["datasetName"] == "FLD":
        fileName = retrainImage.split("_")[1]
        fileName = fileName.split("I")[1]
        fileName = int(fileName.split(".")[0])
        errImage = str(retrainImage.split("_")[1]).split(".")[0] + ".pt"
        if int(caseFile["FLD"]) == 1:
            fileName = str(fileName + 23041)
        if int(caseFile["FLD"]) == 2:
            fileName = str(fileName + 16013)
        fileName = "Test_" + str(fileName)
    else:
        fileName = retrainImage.split("_")[1]
        fileClass = retrainImage.split("_")[0]
        errImage = fileName.split(".")[0] + "_" + fileClass + ".pt"
        fileName = "Test_" + fileName.split(".")[0] + "_" + fileClass
    return errImage, fileName, retrainImage



def assign(trainImage, heatMap):
    global caseFile, errImage, clusterIDX, centroidHMs, centroidRadius
    testHMX, imgList = HeatmapModule.collectHeatmaps(caseFile["filesPath"], caseFile["selectedLayer"])
    #caseFile = torch.load(caseFile["caseFile"])
    print("Assigning..")
    retrainList = caseFile["retrainList"]
    retrainHMPath = join(caseFile["outputPath"], "trainHeatmaps", caseFile["selectedLayer"])
    selection = caseFile["assignMode"]
    selectedLayerClusters = torch.load(caseFile["clsPath"])
    metric = caseFile["metric"]
    datasetName = caseFile["datasetName"]
    area = caseFile["faceSubset"]
    dnn = caseFile["DNN"]
    improveNpy = caseFile["improveDataNpy"]
    imgExt = caseFile["imgExt"]
    breakFlag = False
    if selection == "Centroid":
        trainCentroidDist = []
        for clusterID in clusterIDX:
            trainDist = HeatmapModule.doDistance(centroidHMs[clusterID], heatMap, metric)
            trainCentroidDist.append(trainDist)
        indx = min(enumerate(trainCentroidDist), key=itemgetter(1))[0]

        if centroidRadius[clusterIDX[indx]] - trainCentroidDist[indx] > 0:
            clusterID = clusterIDX[indx]

            sumDistanceWithNewMember = 0
            for testImage in selectedLayerClusters['clusters'][clusterID]['members']:
                sumDistanceWithNewMember += HeatmapModule.doDistance(heatMap, testHM[testImage], metric)

            length = len(selectedLayerClusters['clusters'][clusterID]['members'])
            sumDistance = selectedLayerClusters['clusters'][clusterID]['sumDistance']
            numPairs = ((length * (length - 1)) / 2)
            if length == 1:
                avgDisCandidateCluster = 0
            else:
                avgDisCandidateCluster = sumDistance / numPairs
            sumDisCandidateClusterWitNewMem = sumDistance + sumDistanceWithNewMember
            length = length + 1
            numPairs = ((length * (length - 1)) / 2)
            avgDisCandidateClusterWitNewMem = sumDisCandidateClusterWitNewMem / numPairs

            if avgDisCandidateClusterWitNewMem <= avgDisCandidateCluster:
                if not 'assigned' in selectedLayerClusters['clusters'][clusterID]:
                    selectedLayerClusters['clusters'][clusterID]['assigned'] = []
                selectedLayerClusters['clusters'][clusterID]['assigned'].append(trainImage)
                caseFile["assImages"].append(trainImage)
                caseFile["actualCluster"].append(clusterID)
            else:
                if not 'selected' in selectedLayerClusters['clusters'][clusterID]:
                    selectedLayerClusters['clusters'][clusterID]['selected'] = []
                # print(avgDisCandidateClusterWitNewMem, avgDisCandidateCluster)
                selectedLayerClusters['clusters'][clusterID]['selected'].append(trainImage)
                caseFile["notAssImages"].append(trainImage)
                caseFile["actualCluster"].append(-1)
        else:
            if not 'selected' in selectedLayerClusters['clusters'][clusterIDX[indx]]:
                selectedLayerClusters['clusters'][clusterIDX[indx]]['selected'] = []
            # print(avgDisCandidateClusterWitNewMem, avgDisCandidateCluster)
            selectedLayerClusters['clusters'][clusterIDX[indx]]['selected'].append(trainImage)
            caseFile["notAssImages"].append(trainImage)
            caseFile["actualCluster"].append(-1)

    elif selection == "ClosestICD":
        sumDistanceWithNewMember = {}
        distWithMembers = []
        clusterIDX = []
        for clusterID in selectedLayerClusters['clusters']:
            sumDistanceWithNewMember[clusterID] = 0
            minDist = []
            for testImage in selectedLayerClusters['clusters'][clusterID]['members']:
                Diff = HeatmapModule.doDistance(heatMap, testHM[testImage], metric)
                minDist.append(Diff)
                sumDistanceWithNewMember[clusterID] += Diff
            # indx = min(enumerate(minDist), key=itemgetter(1))[0]
            distWithMembers.append(min(minDist))
            clusterIDX.append(clusterID)
        indx = min(enumerate(distWithMembers), key=itemgetter(1))[0]
        candidateClusterID = clusterIDX[indx]

        if not 'selected' in selectedLayerClusters['clusters'][candidateClusterID]:
            selectedLayerClusters['clusters'][candidateClusterID]['selected'] = []
        length = len(selectedLayerClusters['clusters'][candidateClusterID]['members'])
        if length == 1:
            avgDisCandidateCluster = 0
        else:
            avgDisCandidateCluster = selectedLayerClusters['clusters'][candidateClusterID][
                                         'sumDistance'] / (
                                             (length * (length - 1)) / 2)
        sumDisCandidateClusterWitNewMem = selectedLayerClusters['clusters'][candidateClusterID][
                                              'sumDistance'] + sumDistanceWithNewMember[candidateClusterID]
        avgDisCandidateClusterWitNewMem = sumDisCandidateClusterWitNewMem / (((length + 1) * ((length + 1) - 1)) / 2)
        if not 'assigned' in selectedLayerClusters['clusters'][candidateClusterID]:
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'] = []
        # if avgDisCandidateClusterWitNewMem <= (avgDisCandidateCluster + (avgDisCandidateCluster*0.01)): # + 1%
        if avgDisCandidateClusterWitNewMem <= (avgDisCandidateCluster):  # + 0%
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'].append(trainImage)
            caseFile["assImages"].append(trainImage)
            caseFile["actualCluster"].append(candidateClusterID)
        else:
            caseFile["notAssImages"].append(trainImage)
            selectedLayerClusters['clusters'][candidateClusterID]['selected'].append(trainImage)
            caseFile["actualCluster"].append(-1)

    elif selection == "jICD":
        deltaICD = {}
        for clusterID in selectedLayerClusters['clusters']:
            sumDistanceWithNewMember = 0
            for testImage in selectedLayerClusters['clusters'][clusterID]['members']:
                sumDistanceWithNewMember += HeatmapModule.doDistance(heatMap, testHM[testImage], metric)
            sumDistance = selectedLayerClusters['clusters'][clusterID]['sumDistance']
            length = len(selectedLayerClusters['clusters'][clusterID]['members'])
            numPairs = ((length * (length - 1)) / 2)
            if numPairs == 0:
                deltaICD[clusterID] = -1
            else:
                oldICD = sumDistance / numPairs
                # oldICD = (oldICD*0.05) + oldICD
                # oldICD = (oldICD*0.01) + oldICD
                length = length + 1
                numPairs = ((length * (length - 1)) / 2)
                newICD = (sumDistance + sumDistanceWithNewMember) / numPairs
                deltaICD[clusterID] = oldICD - newICD
        candidateClusterID = max(deltaICD.keys(), key=(lambda k: deltaICD[k]))
        if not 'assigned' in selectedLayerClusters['clusters'][candidateClusterID]:
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'] = []
            selectedLayerClusters['clusters'][candidateClusterID]['selected'] = []

        length = len(selectedLayerClusters['clusters'][candidateClusterID]['members'])
        if length == 1:
            caseFile["notAssImages"].append(trainImage)
            caseFile["actualCluster"].append(-1)
        else:
            if deltaICD[candidateClusterID] >= 0:
                selectedLayerClusters['clusters'][candidateClusterID]['assigned'].append(trainImage)
                caseFile["assImages"].append(trainImage)
                caseFile["actualCluster"].append(candidateClusterID)
            else:
                caseFile["notAssImages"].append(trainImage)
                caseFile["actualCluster"].append(-1)

    elif selection == 'ClosestMem':
        distWithMembers = []
        clusterIDX = []
        for clusterID in selectedLayerClusters['clusters']:
            minDist = []
            for testImage in selectedLayerClusters['clusters'][clusterID]['members']:
                Diff = HeatmapModule.doDistance(heatMap, testHM[testImage], metric)
                minDist.append(Diff)
            # indx = min(enumerate(minDist), key=itemgetter(1))[0]
            distWithMembers.append(min(minDist))
            clusterIDX.append(clusterID)

        indx = min(enumerate(distWithMembers), key=itemgetter(1))[0]
        candidateClusterID = clusterIDX[indx]
        if not 'selected' in selectedLayerClusters['clusters'][candidateClusterID]:
            selectedLayerClusters['clusters'][candidateClusterID]['selected'] = []
        if not 'assigned' in selectedLayerClusters['clusters'][candidateClusterID]:
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'] = []
        selectedLayerClusters['clusters'][candidateClusterID]['assigned'].append(trainImage)
        caseFile["assImages"].append(trainImage)
        caseFile["actualCluster"].append(candidateClusterID)

    elif selection == 'Entropy':
        improveRCCDists = caseFile["improveRCCDists"]
        if not isfile(join(improveRCCDists, "closestClusterDist.pt")):
            closestClusterDist = {}
            torch.save(closestClusterDist, join(improveRCCDists, "closestClusterDist.pt"))
        else:
            closestClusterDist = torch.load(join(improveRCCDists, "closestClusterDist.pt"))
        x = 0
        print("\n")
        loadBar = 0
        RemainingTime = "N/A"
        start = time.time()
        random.shuffle(retrainList)
        if len(closestClusterDist) < len(retrainList):
            for trainImage in retrainList:
                if trainImage not in closestClusterDist:
                    heatMap, E = HeatmapModule.generateHeatMap(trainImage, dnn, datasetName, "", False, area, improveNpy
                                                               , imgExt, caseFile["FLD"])
                    closestClusterDist[trainImage] = E
                x += 1
                if x / int(len(retrainList) * 0.01) == 1:
                    loadBar += 1.0
                    spentTime = ((time.time() - start) / 60.0)
                    timePerLoadBar = spentTime/loadBar
                    spentTime = timePerLoadBar * loadBar
                    fullTime = timePerLoadBar * 100
                    remTime = math.ceil(fullTime - spentTime)
                    if remTime > 60:
                        RemainingTime = str(remTime/60)[0:4] + "hs."
                    else:
                        RemainingTime = str(remTime) + " mins."
                    x = 0
                    try:
                        closestClusterDist_local = torch.load(join(improveRCCDists, "closestClusterDist.pt"))
                        for img in closestClusterDist_local:
                            if img not in closestClusterDist:
                                closestClusterDist[img] = closestClusterDist_local[img]
                    except EOFError as error:
                        print("EOFError")
                    except TypeError as error:
                        print("TypeError")
                    torch.save(closestClusterDist, join(improveRCCDists, "closestClusterDist.pt"))
                else:
                    print("Checked:", str(loadBar) + "%", "ETA: <" + str(RemainingTime), "Collected: " +
                          str(int(100.00 * len(closestClusterDist)/len(retrainList))) + "%", end="\r")
        assignedList = []
        orderLength = len(selectedLayerClusters['clusters']) * len(retrainList)
        start = time.time()
        loadBar = 0
        x = 0
        clusterUCs, totalAssigned, totalUc, totalUb, Ub = DS.getUCs(caseFile, 2)
        toAssign = totalUc
        for _ in retrainList:
            candidateImage = max(closestClusterDist, key=closestClusterDist.get)
            E = closestClusterDist[candidateImage]
            closestClusterDist[candidateImage] = 1e9
            if assignedList.count(candidateImage) < 1:
                if E != 1e9:
                    if len(assignedList) < toAssign:
                        assignedList.append(candidateImage)
                        caseFile["assImages"] = assignedList
                else:
                    break
            x += 1
            if x / (orderLength * 0.01) == 1:
                loadBar += 1.0
                spentTime = ((time.time() - start) / 60.0)
                timePerLoadBar = spentTime/loadBar
                spentTime = timePerLoadBar * loadBar
                fullTime = timePerLoadBar * 100
                remTime = math.ceil(fullTime - spentTime)
                if remTime > 60:
                    RemainingTime = str(remTime/60)[0:4] + "hs."
                else:
                    RemainingTime = str(remTime) + " mins."
                x = 0
            else:
                print("Checked:", str(loadBar) + "%", " ETA: <" + str(RemainingTime),
                      "Assigned: ", str(int(100.00 * len(assignedList) / toAssign)) + "%", end="\r")
            if len(assignedList) == toAssign:
                break
        for _ in retrainList:
            candidateClusterID = -1
            caseFile["actualCluster"].append(candidateClusterID)
        torch.save(caseFile, caseFile["caseFile"])
        breakFlag = True

    elif selection == 'ClosestU':
        improveRCCDists = caseFile["improveRCCDists"]
        if not isfile(join(improveRCCDists, "closestClusterName.pt")):
            closestClusterName = {}
            for order in range(0, len(selectedLayerClusters['clusters'])):
                closestClusterName[order] = {}
                torch.save(closestClusterName, join(improveRCCDists, "closestClusterName.pt"))
        else:
            closestClusterName = torch.load(join(improveRCCDists, "closestClusterName.pt"))
        if not isfile(join(improveRCCDists, "closestClusterDist.pt")):
            closestClusterDist = {}
            for order in range(0, len(selectedLayerClusters['clusters'])):
                closestClusterDist[order] = {}
                torch.save(closestClusterDist, join(improveRCCDists, "closestClusterDist.pt"))
        else:
            closestClusterDist = torch.load(join(improveRCCDists, "closestClusterDist.pt"))
        if not isfile(join(improveRCCDists, "imagesEntropy.pt")):
            imagesEntropy = {}
            torch.save(imagesEntropy, join(improveRCCDists, "imagesEntropy.pt"))
        else:
            imagesEntropy = torch.load(join(improveRCCDists, "imagesEntropy.pt"))
        clustCounter = {}
        x = 0
        dictResult = {}
        print("\n")
        loadBar = 0
        RemainingTime = "N/A"
        start = time.time()
        random.shuffle(retrainList)
        layerIndex = int(caseFile["selectedLayer"].replace("Layer", ""))
        if len(closestClusterDist[0]) < len(retrainList):
            for trainImage in retrainList:
                if trainImage not in closestClusterDist[0]:
                    heatMap, E = HeatmapModule.generateHeatMap(trainImage, dnn, datasetName, "", False, area, improveNpy
                                                               , imgExt, caseFile["FLD"])
                    imagesEntropy[trainImage] = E
                    #errImagex, fileName, retrainImage = nameMapper(trainImage)
                    #heatMap, E = HeatmapModule.safeHM(join(retrainHMPath, errImagex), caseFile["layerIndex"],
                    #                               trainImage, dnn, datasetName, "", False, area, improveNpy, imgExt,
                    #                               caseFile["FLD"])
                    clusterIDX = []
                    clusterDists = []
                    for clusterID in selectedLayerClusters['clusters']:
                        if len(selectedLayerClusters['clusters'][clusterID]['members']) > 1:
                            dictResult[clusterID] = []
                            minDist = []
                            for testImage in selectedLayerClusters['clusters'][clusterID]['members']:
                                Diff = HeatmapModule.doDistance(testHMX[testImage], heatMap[layerIndex], metric)
                                minDist.append(Diff)
                            clusterDists.append(min(minDist))
                            clusterIDX.append(clusterID)
                    if len(clusterDists) > 0:
                        for order in range(0, len(selectedLayerClusters['clusters'])):
                            indx = min(enumerate(clusterDists), key=itemgetter(1))[0]
                            closestClusterDist[order][trainImage] = clusterDists[indx]
                            closestClusterName[order][trainImage] = clusterIDX[indx]
                            clusterDists[indx] = 1e9
                x += 1
                if x / int(len(retrainList) * 0.01) == 1:
                    loadBar += 1.0
                    spentTime = ((time.time() - start) / 60.0)
                    timePerLoadBar = spentTime/loadBar
                    spentTime = timePerLoadBar * loadBar
                    fullTime = timePerLoadBar * 100
                    remTime = math.ceil(fullTime - spentTime)
                    if remTime > 60:
                        RemainingTime = str(remTime/60)[0:4] + "hs."
                    else:
                        RemainingTime = str(remTime) + " mins."
                    x = 0
                    try:
                        closestClusterDist_local = torch.load(join(improveRCCDists, "closestClusterDist.pt"))
                        closestClusterName_local = torch.load(join(improveRCCDists, "closestClusterName.pt"))
                        for img in closestClusterDist_local[0]:
                            if img not in closestClusterDist[0]:
                                for order in range(0, len(selectedLayerClusters['clusters'])):
                                    closestClusterDist[order][img] = closestClusterDist_local[order][img]
                                    closestClusterName[order][img] = closestClusterName_local[order][img]
                    except EOFError as error:
                        print("EOFError")
                    except TypeError as error:
                        print("TypeError")
                    torch.save(closestClusterDist, join(improveRCCDists, "closestClusterDist.pt"))
                    torch.save(closestClusterName, join(improveRCCDists, "closestClusterName.pt"))
                    torch.save(imagesEntropy, join(improveRCCDists, "imagesEntropy.pt"))
                else:
                    print("Checked:", str(loadBar) + "%", "ETA: <" + str(RemainingTime), "Collected: " +
                          str(int(100.00 * len(closestClusterDist[0])/len(retrainList))) + "%", end="\r")
        assignedList = []
        orderLength = len(selectedLayerClusters['clusters']) * len(retrainList)
        start = time.time()
        loadBar = 0
        x = 0
        clusterUCs, totalAssigned, totalUc, totalUb, Ub = DS.getUCs(caseFile, 2)
        toAssign = totalUc
        for clusterID in selectedLayerClusters['clusters']:
            if not 'selected' in selectedLayerClusters['clusters'][clusterID]:
                selectedLayerClusters['clusters'][clusterID]['selected'] = []
            if not 'assigned' in selectedLayerClusters['clusters'][clusterID]:
                selectedLayerClusters['clusters'][clusterID]['assigned'] = []
            clustCounter[clusterID] = 0
            dictResult[clusterID] = []
        entropyList = list()
        for _ in imagesEntropy:
            candidateImage = max(imagesEntropy.keys(), key=(lambda k: imagesEntropy[k]))
            imagesEntropy[candidateImage] = 0
            entropyList.append(candidateImage)
        entropyList = entropyList[0:math.ceil(totalAssigned)]
        for order in range(0, len(selectedLayerClusters['clusters'])):
            for _ in closestClusterDist[order]:
                candidateImage = min(closestClusterDist[order].keys(), key=(lambda k: closestClusterDist[order][k]))
                candidateClusterID = closestClusterName[order][candidateImage]
                distance = closestClusterDist[order][candidateImage]
                closestClusterDist[order][candidateImage] = 1000
                if caseFile["retrainMode"] == "HUDDE":
                    if entropyList.count(candidateImage) == 0:
                        continue
                if assignedList.count(candidateImage) < 1:
                    if distance < 1000:
                        if dictResult[candidateClusterID].count(candidateImage) < 1:
                            if len(dictResult[candidateClusterID]) < clusterUCs[candidateClusterID]:
                                    dictResult[candidateClusterID].append(candidateImage)
                                    selectedLayerClusters['clusters'][candidateClusterID]['assigned'].append(candidateImage)
                                    assignedList.append(candidateImage)
                                    caseFile["assImages"] = assignedList
                                    clustCounter[candidateClusterID] += 1
                    else:
                        break
                x += 1
                if x / (orderLength * 0.01) == 1:
                    loadBar += 1.0
                    spentTime = ((time.time() - start) / 60.0)
                    timePerLoadBar = spentTime/loadBar
                    spentTime = timePerLoadBar * loadBar
                    fullTime = timePerLoadBar * 100
                    remTime = math.ceil(fullTime - spentTime)
                    if remTime > 60:
                        RemainingTime = str(remTime/60)[0:4] + "hs."
                    else:
                        RemainingTime = str(remTime) + " mins."
                    x = 0
                else:
                    print("Checked:", str(loadBar) + "%", " ETA: <" + str(RemainingTime),
                          "Assigned: ", str(int(100.00 * len(assignedList) / toAssign)) + "%", "order:", order, end="\r")
                if len(assignedList) == toAssign:
                    break
        for trainImage in retrainList:
            candidateClusterID = -1
            for clusterID in dictResult:
                if dictResult[clusterID].count(trainImage) > 0:
                    candidateClusterID = clusterID
            caseFile["actualCluster"].append(candidateClusterID)
        torch.save(caseFile, caseFile["caseFile"])
        #print("Identical Assigned Cluster Images:", y)
        #print("Total Identical:", hh)
        breakFlag = True

    elif selection == "SSEICD":
        sumSquareWithNewMember = {}
        sumDistanceWithNewMember = {}
        errorSumWithNewMember = {}
        diffSumWithNewMember = {}
        for clusterID in selectedLayerClusters['clusters']:
            errorSumWithNewMember[clusterID] = 0
            sumDistanceWithNewMember[clusterID] = 0
            sumSquareWithNewMember[clusterID] = 0
            for testImage in selectedLayerClusters['clusters'][clusterID]['members']:
                sumSquareWithNewMember[clusterID] += HeatmapModule.doDistance(heatMap, testHM[testImage], metric) ** 2
                sumDistanceWithNewMember[clusterID] += HeatmapModule.doDistance(heatMap, testHM[testImage], metric)
            errorSumWithNewMember[clusterID] = (sumSquareWithNewMember[clusterID] +
                                                selectedLayerClusters['clusters'][clusterID]['variance']) / (len(
                selectedLayerClusters['clusters'][clusterID]['members']) + 1)
            diffSumWithNewMember[clusterID] = errorSumWithNewMember[clusterID] - \
                                              selectedLayerClusters['clusters'][clusterID]['errorSum']

        candidateClusterID = min(diffSumWithNewMember.keys(), key=(lambda k: diffSumWithNewMember[k]))
        length = len(selectedLayerClusters['clusters'][candidateClusterID]['members'])
        if length == 1:
            avgDisCandidateCluster = 0
        else:
            avgDisCandidateCluster = selectedLayerClusters['clusters'][candidateClusterID]['sumDistance'] / (
                    (length * (length - 1)) / 2)
        sumDisCandidateClusterWitNewMem = selectedLayerClusters['clusters'][candidateClusterID]['sumDistance'] + \
                                          sumDistanceWithNewMember[candidateClusterID]
        avgDisCandidateClusterWitNewMem = sumDisCandidateClusterWitNewMem / (((length + 1) * ((length + 1) - 1)) / 2)
        if not 'assigned' in selectedLayerClusters['clusters'][candidateClusterID]:
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'] = []
            selectedLayerClusters['clusters'][candidateClusterID]['selected'] = []
        if avgDisCandidateClusterWitNewMem <= avgDisCandidateCluster:
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'].append(trainImage)
            caseFile["assImages"].append(trainImage)
            caseFile["actualCluster"].append(candidateClusterID)
        else:
            selectedLayerClusters['clusters'][candidateClusterID]['selected'].append(trainImage)
            caseFile["notAssImages"].append(trainImage)
            caseFile["actualCluster"].append(-1)

    elif selection == 'jSSE':

        sumSquareWithNewMember = {}
        sumDistanceWithNewMember = {}
        errorSumWithNewMember = {}
        diffSumWithNewMember = {}
        for clusterID in selectedLayerClusters['clusters']:
            errorSumWithNewMember[clusterID] = 0
            sumDistanceWithNewMember[clusterID] = 0
            sumSquareWithNewMember[clusterID] = 0
            for testImage in selectedLayerClusters['clusters'][clusterID]['members']:
                sumSquareWithNewMember[clusterID] += HeatmapModule.doDistance(heatMap, testHM[testImage], metric) ** 2
                sumDistanceWithNewMember[clusterID] += HeatmapModule.doDistance(heatMap, testHM[testImage], metric)
            errorSumWithNewMember[clusterID] = (sumSquareWithNewMember[clusterID] +
                                                selectedLayerClusters['clusters'][clusterID]['variance']) / (len(
                selectedLayerClusters['clusters'][clusterID]['members']) + 1)
            diffSumWithNewMember[clusterID] = errorSumWithNewMember[clusterID] - \
                                              selectedLayerClusters['clusters'][clusterID]['errorSum']

        candidateClusterID = min(diffSumWithNewMember.keys(), key=(lambda k: diffSumWithNewMember[k]))
        if not 'assigned' in selectedLayerClusters['clusters'][candidateClusterID]:
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'] = []
            selectedLayerClusters['clusters'][candidateClusterID]['selected'] = []
        if diffSumWithNewMember[candidateClusterID] <= 0:
            selectedLayerClusters['clusters'][candidateClusterID]['assigned'].append(trainImage)
            caseFile["assImages"].append(trainImage)
            caseFile["actualCluster"].append(candidateClusterID)
        else:
            selectedLayerClusters['clusters'][candidateClusterID]['selected'].append(trainImage)
            caseFile["notAssImages"].append(trainImage)
            caseFile["actualCluster"].append(-1)

    #elif selection == 'HMEntropy':

    torch.save(caseFile, caseFile["caseFile"])
    return selectedLayerClusters, breakFlag


def calculate_pixel_distance(coord1, coord2):
    diff = np.square(coord1 - coord2)
    sum_diff = np.sqrt(diff[:, :, 0] + diff[:, :, 1])
    avg = sum_diff.mean()
    return avg, sum_diff


def saveRetrainHM():
    global caseFile
    model = caseFile["DNN"]
    print("Saving retrain Heatmaps")
    if caseFile["datasetName"] == "FLD":
        KParray = getKParray()
        counter = 1
        index = 0
        makeFolder(caseFile["outputPath"])
        dataset = np.load(caseFile["improveDataNpy"], allow_pickle=True)
        dataset = dataset.item()
        x_data = dataset["data"]
        x_data = x_data.astype(np.float32)
        x_data = x_data / 255.
        x_data = x_data[:, np.newaxis]
        for inputs in x_data:
            imageName = "I" + str(counter) + ".pt"
            savePath = join(caseFile["outputPath"], imageName)
            if not isfile(savePath):
                transformer = setupTransformer(caseFile["datasetName"])
                inputs = transformer(inputs)
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    model = model.cuda()
                else:
                    inputs = Variable(inputs)
                model = HeatmapModule.ieeRegister(model)
                predict = model(inputs.unsqueeze(0).float())
                predict_cpu = predict.cpu()
                predict_cpu = predict_cpu.detach().numpy()
                predict_xy1 = DS.transfer_target(predict_cpu, n_points=DS.n_points)
                # labels_gt = dataset["label"][0]
                # print(labels_gt)
                labels_gt = dataset["label"][index]
                labels_msk = np.ones(labels_gt.shape)
                labels_msk[labels_gt <= 1e-5] = 0
                predict_xy = np.multiply(predict_xy1, labels_msk)
                avg, sum_diff = calculate_pixel_distance(labels_gt, predict_xy)
                label = 0
                worst_label = 0
                worst_KP = 0
                for KP in sum_diff[0]:
                    if KParray.count(label) > 0:
                        if KP > worst_KP:
                            worst_KP = KP
                            worst_label = label
                    label += 1
                # print(worst_label)
                # KPindex = int(row["worst_KP"][2::])
                predict_cpu = HeatmapModule.ieeBackKP(predict_cpu, worst_label)
                # predict_cpu = HeatmapModule.ieeBackParts(predict_cpu, area)
                tAF = torch.from_numpy(predict_cpu[0]).type(torch.FloatTensor)
                if torch.cuda.is_available():
                    tAF = Variable(tAF).cuda()
                else:
                    tAF = Variable(tAF).cpu()
                model.relprop(tAF)
                heatmaps = HeatmapModule.returnHeatmap(model)
                torch.save(heatmaps[caseFile["layerIndex"]], savePath)
                del heatmaps
            counter += 1
            index += 1
            if counter % 10000 == 0:
                print("Checked {} images".format(counter), end="\r")
    else:
        counter = 0
        retrainList, retrainLength = getFolderSize(caseFile["improveDataPath"])
        fileList = retrainList
        random.shuffle(fileList)
        for file in fileList:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".ppm"):
                fileName = str(basename(file).split(".")[0]) + ".pt"
                savePath = join(caseFile["outputPath"], basename(dirname(file)) + "_" + fileName)
                filePath = file
                # heatmaps = generateActivations(inputImage, model, datasetName, outputPath, True)
                if not isfile(savePath):
                    heatmaps = HeatmapModule.generateHeatMap(filePath, caseFile["DNN"], caseFile["datasetName"],
                                                             caseFile["outputPath"], False, caseFile["imgExt"])
                    torch.save(heatmaps[caseFile["layerIndex"]], savePath)
                    del heatmaps
                counter = counter + 1
                if counter % 10000 == 0:
                    print("Checked and Saved " + str(counter) + " improvement images.")
    print("Checked and Saved " + str(counter) + " improvement images.")
    return


def getFolderSize(inputPath):
    Counter = 0
    imgList = []
    for src_dir, dirs, files in os.walk(inputPath):
        for file in files:
            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".ppm")):
                Counter = Counter + 1
                imgList.append(join(src_dir, file))
    return imgList, Counter


def makeFolder(inputPath):
    if not exists(inputPath):
        os.makedirs(inputPath)


def classifyImprov(datasetName, dnn, fileName, filePath, labelPath, CC, MC):
    resStr = testModule.testDNN(datasetName, dnn, fileName, filePath, labelPath)
    if resStr == 'M':
        MC.append(fileName)
    else:
        CC.append(fileName)
    return CC, MC


def getClusterData(caseFile, heatmapsDistance):
    global testHM, centroidHMs, centroidRadius, clusterIDX
    if torch.cuda.is_available():
        selectedLayerClusters = torch.load(caseFile["clsPath"])
    else:
        selectedLayerClusters = torch.load(caseFile["clsPath"], map_location=torch.device('cpu'))
    metric = caseFile["metric"]
    layer = caseFile["selectedLayer"]
    print("Collecting cluster data")
    centroidHMs = {}
    medoidHMs = {}
    centroidRadius = {}
    medoidRadius = {}
    clusterIDX = []
    testHM = {}
    cls = len(selectedLayerClusters['clusters'])
    cls2 = 0
    testHM, imgList = HeatmapModule.collectHeatmaps(caseFile["filesPath"], layer)
    for clusterID in selectedLayerClusters['clusters']:
        SSE = 0
        cls2 += 1
        sumDistance = selectedLayerClusters['clusters'][clusterID]['sumDistance']
        length = len(selectedLayerClusters['clusters'][clusterID]['members'])
        numPairs = ((length * (length - 1)) / 2)
        print(str(cls2 / cls * 100.00)[0:5] + "%", end="\r")
        if not 'Distances' in selectedLayerClusters['clusters'][clusterID]:
            selectedLayerClusters['clusters'][clusterID]['Distances'] = []
        if numPairs == 0:
            selectedLayerClusters['clusters'][clusterID]['ICD'] = 0
        else:
            selectedLayerClusters['clusters'][clusterID]['ICD'] = sumDistance / numPairs
        clusterMember = selectedLayerClusters['clusters'][clusterID]['members']
        i = 0
        for errImage in os.listdir(join(caseFile["filesPath"], "Heatmaps", layer)):
            errImage = errImage.split(".")[0]
            if i == 0:
                sumHM = torch.add(testHM[errImage], 0)
            else:
                sumHM = torch.add(testHM[errImage], sumHM)
            i = i + 1
        centroidHMs[clusterID] = torch.div(sumHM, length)
        #if 'SSE' not in selectedLayerClusters['clusters'][clusterID]:
        if 'Medoid-Farthest-Dist' not in selectedLayerClusters['clusters'][clusterID]:
        #if True:
            selectedLayerClusters['clusters'][clusterID]['Centroid-HM'] = centroidHMs[clusterID]
            maxDist = 0
            maxDist2 = 0
            minDist = 1e9
            maxRadius = 0
            minRadius = 1e9
            maxCentMember = clusterMember[0]
            maxMedMember = clusterMember[0]
            minCentMember = clusterMember[0]
            maxMember1 = clusterMember[0]
            maxMember2 = clusterMember[0]
            minMember1 = clusterMember[0]
            minMember2 = clusterMember[0]
            heatMapDistanceExecl = pd.read_excel(heatmapsDistance)
            heatMapDistanceExecl.drop(heatMapDistanceExecl.columns[heatMapDistanceExecl.columns.str.contains('unnamed',
                                                                                                         case=False)],
                                  axis=1, inplace=True)
            #heatMapDistanceExecl = heatmapsDistance
            distance = heatMapDistanceExecl.values
            clusterMembersName = heatMapDistanceExecl.columns
            distDict = {}
            for m1 in range(0, len(clusterMember)):
                list_ = []
                for m2 in range(0, len(clusterMember)):
                    if m1 != m2:
                        indexM1 = clusterMembersName.get_loc(clusterMember[m1])
                        indexM2 = clusterMembersName.get_loc(clusterMember[m2])
                        dist = distance[indexM1][indexM2]
                        list_.append(dist)
                distDict[str(m1)] = sum(list_)/len(list_)
            newDict = dict(sorted(distDict.items(), key=lambda item: item[1]))
            members = []
            #medoidMember = clusterMember[int(newDict[0])] #1st medoid
            #medoidMember = clusterMember[int(newDict[1])] #2nd medoid
            for medoidNumber in newDict:
                members.append(clusterMember[int(medoidNumber)])
            medoidMember = members[0]

            medoidHMs[clusterID] = testHM[medoidMember]
            for m1 in range(0, len(clusterMember)):
                indexM1 = clusterMembersName.get_loc(clusterMember[m1])
                Diff = HeatmapModule.doDistance(centroidHMs[clusterID], testHM[clusterMember[m1]], metric)
                Diff2 = HeatmapModule.doDistance(medoidHMs[clusterID], testHM[clusterMember[m1]], metric)
                if Diff2 > maxDist2:
                    maxDist2 = Diff2
                    maxMedMember = clusterMember[m1]
                if Diff > maxDist:
                    maxDist = Diff
                    maxCentMember = clusterMember[m1]
                if Diff < minDist:
                    minDist = Diff
                    minCentMember = clusterMember[m1]
                for m2 in range(m1 + 1, len(clusterMember)):
                    indexM2 = clusterMembersName.get_loc(clusterMember[m2])
                    dist = distance[indexM1][indexM2]
                    SSE += dist ** 2
                    selectedLayerClusters['clusters'][clusterID]['Distances'].append(dist)
                    if dist > maxRadius:
                        maxRadius = dist
                        maxMember1 = clusterMember[m1]
                        maxMember2 = clusterMember[m2]
                    if dist < minRadius:
                        minRadius = dist
                        minMember1 = clusterMember[m1]
                        minMember2 = clusterMember[m2]
            selectedLayerClusters['clusters'][clusterID]['SSE'] = SSE
            selectedLayerClusters['clusters'][clusterID]['SSE/Len'] = SSE / length
            selectedLayerClusters['clusters'][clusterID]['minMem1'] = minMember1
            selectedLayerClusters['clusters'][clusterID]['minMem2'] = minMember2
            selectedLayerClusters['clusters'][clusterID]['maxMem1'] = maxMember1
            selectedLayerClusters['clusters'][clusterID]['maxMem2'] = maxMember2
            selectedLayerClusters['clusters'][clusterID]['Centroid-Farthest-Dist'] = maxDist
            selectedLayerClusters['clusters'][clusterID]['Centroid-Farthest-Member'] = maxCentMember
            selectedLayerClusters['clusters'][clusterID]['Centroid-Closest-Dist'] = minDist
            selectedLayerClusters['clusters'][clusterID]['Centroid-Closest-Member'] = minCentMember
            selectedLayerClusters['clusters'][clusterID]['Medoid-Member'] = medoidMember
            selectedLayerClusters['clusters'][clusterID]['Medoid-Farthest-Dist'] = maxDist2
            selectedLayerClusters['clusters'][clusterID]['Medoid-Farthest-Member'] = maxMedMember
        centroidRadius[clusterID] = selectedLayerClusters['clusters'][clusterID]['Centroid-Farthest-Dist']
        medoidRadius[clusterID] = selectedLayerClusters['clusters'][clusterID]['Medoid-Farthest-Dist']
        medoidHMs[clusterID] = testHM[selectedLayerClusters['clusters'][clusterID]['Medoid-Member']]
        clusterIDX.append(clusterID)
        print(clusterID, selectedLayerClusters['clusters'][clusterID]['Medoid-Member'], selectedLayerClusters['clusters'][clusterID]['Medoid-Farthest-Member'])
    torch.save(selectedLayerClusters, caseFile["clsPath"])
    #return centroidRadius, centroidHMs, testHM
    return medoidRadius, medoidHMs, testHM


def getKParray():
    global caseFile
    area = caseFile["faceSubset"]
    rightbrow = [2, 3]
    leftbrow = [0, 1]
    mouth = [23, 24, 25, 26]
    righteye = [16, 17, 18, 19, 20, 21, 22]
    lefteye = [9, 10, 11, 12, 13, 14, 15]
    noseridge = [4, 5]
    nose = [6, 7, 8]
    KParray = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
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
    return KParray