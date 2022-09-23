#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#
import testModule
from imports import setupTransformer, os, Image, Variable, argparse, datasets, torch, pd, np, isfile, basename, join, tf

import dataSupplier as dataSupply
import testModule
import dnnModels
import ieepredict
from imports import sys, torch, nn, optim, Variable, np, shutil, os, time, datasets, math, pd, PathImageFolder, \
    setupTransformer, exists, join, isfile, isdir, basename, dirname

from dataSupplier import DataSupplier
from imports import np, optim, torch, Variable, math
import time

learning_rate = 0.001
momentum = 0.9
log_schedule = 10
nPoints = 64

import scipy.misc

BATCH_SIZE = 100
DATA_DIR = '/vol/data'

LOGDIR = './train_model'
CHECKPOINT_EVERY = 100
NUM_STEPS = int(1e5)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 1e-3
KEEP_PROB = 0.8
L2_REG = 0
EPSILON = 0.001
MOMENTUM = 0.9


def run(caseFile_):
    global caseFile
    global expIndex
    global BaggedUnsafeSet
    caseFile = caseFile_
    caseFile["DNNsPath"] = join(caseFile["outputPathOriginal"], caseFile["RCC"], "DNNModels_")
    caseFile[caseFile["retrainMode"]] = {}
    start = time.time()
    if caseFile_["Alex"]:
        avgImp, maxImp, minImp, avgWor, maxWor, minWor, avgTestSet, DNNModels, dumbDict, clusterDict = retrainAlex()
    if caseFile_["KP"]:
        # avgImp, maxImp, minImp, avgWor, maxWor, minWor, avgTestSet, DNNModels, dumbDict, clusterDict = retrainKP()
        retrainKP()
        caseFile[caseFile["retrainMode"]]["failCount"] = caseFile[caseFile["retrainMode"]]["totalFailCount"] = 0
    if "retrieveAccuracy" not in caseFile:
        failCount = caseFile[caseFile["retrainMode"]]["failCount"] / (caseFile["expNum2"] - caseFile["expNum1"] + 1)
        failCount_ = 100.00 * (failCount / BaggedUnsafeSet)
        totalFailCount = caseFile[caseFile["retrainMode"]]["totalFailCount"] / (
                caseFile["expNum2"] - caseFile["expNum1"] + 1)
        totalFailCount_ = 100.00 * (totalFailCount / BaggedUnsafeSet)
    else:
        dumbDict = torch.load(join(DNNModels, caseFile["retrainMode"] + "_resultDict.pt"))
        failCount = dumbDict["failCount"]
        failCount_ = dumbDict["fail%"]
    caseFile[caseFile["retrainMode"]]["AvgImproved"] = avgImp
    caseFile[caseFile["retrainMode"]]["MaxImproved"] = maxImp
    caseFile[caseFile["retrainMode"]]["MinImproved"] = minImp
    caseFile[caseFile["retrainMode"]]["AvgWorsened"] = avgWor
    caseFile[caseFile["retrainMode"]]["MaxWorsened"] = maxWor
    caseFile[caseFile["retrainMode"]]["MinWorsened"] = minWor
    caseFile[caseFile["retrainMode"]]["AvgTestAccuracy"] = avgTestSet
    caseFile[caseFile["retrainMode"]]["failCount"] = failCount
    caseFile[caseFile["retrainMode"]]["totalFailCount"] = totalFailCount
    print(caseFile["retrainMode"], str(caseFile["expNum2"] - caseFile["expNum1"] + 1) + " exp.", str(avgTestSet) + "%")
    print("Improved (avg/min/max):", str(avgImp) + "/" + str(minImp) + "/" + str(maxImp))
    print("Worsened (avg/min/max):", str(avgWor) + "/" + str(minWor) + "/" + str(maxWor))
    # print("Failing% (avg):", str(failCount_))
    # print("Total Failing% (avg):", str(totalFailCount_))
    clsWithAssImages = torch.load(caseFile["clsPath"])
    for clusterID in clusterDict:
        if clusterID in clsWithAssImages['clusters']:
            clusterDict[clusterID]["Imp"] /= (caseFile["expNum2"] - caseFile["expNum1"] + 1)
            clusterDict[clusterID]["Imp"] /= len(clsWithAssImages['clusters'][clusterID]['members'])
            clusterDict[clusterID]["Imp"] *= 100.00
            clusterDict[clusterID]["Wor"] /= (caseFile["expNum2"] - caseFile["expNum1"] + 1)
            clusterDict[clusterID]["Wor"] /= len(clsWithAssImages['clusters'][clusterID]['members'])
            clusterDict[clusterID]["Wor"] *= 100.00
    # print("Avg % of Improved Images per Cluster:")
    avgImpPerCls = list()
    for i in range(1, len(clusterDict) + 1):
        # print(i, str(clusterDict[i]["Imp"])[0:5])
        avgImpPerCls.append(clusterDict[i]["Imp"])
    # print("Avg % of Improved Images per Cluster:", sum(avgImpPerCls)/len(avgImpPerCls))
    dumbDict["Improved"] = str(avgImp) + "/" + str(minImp) + "/" + str(maxImp)
    dumbDict["Worsened"] = str(avgWor) + "/" + str(minWor) + "/" + str(maxWor)
    dumbDict["AvgTestAccuracy"] = avgTestSet
    dumbDict["failCount"] = failCount
    dumbDict["fail%"] = failCount_
    torch.save(dumbDict, join(DNNModels, caseFile["retrainMode"] + "_resultDict.pt"))
    torch.save(caseFile, caseFile["caseFile"])
    newName = str(caseFile["retrainMode"]) + "_" + str(caseFile[caseFile["retrainMode"]]["AvgTestAccuracy"])[0:6]
    oldPath = caseFile["DNNsPath"] + str(caseFile["retrainMode"]) + "_" + expIndex
    newPath = caseFile["DNNsPath"] + newName
    if "retrieveAccuracy" not in caseFile:
        shutil.copytree(oldPath, newPath)
        shutil.rmtree(oldPath)
    torch.save(caseFile, join(newPath, "caseFile.pt"))
    print("Total time of batch job is " + str((time.time() - start) / 60.0) + " minutes.")
    print("*****************************")
    print("*****************************")
    print("*****************************")


def retrainKP():
    expNumber = caseFile["expNum1"]
    expNumber2 = caseFile["expNum2"]
    outputPathOriginal = caseFile["outputPathOriginal"]
    retrainMode = caseFile["retrainMode"]
    batchSize = caseFile["batchSize"]
    Epochs = caseFile["Epochs"]
    components = caseFile["components"]
    modelsPath = join(outputPathOriginal, "DNNModels")
    realDataNpy = caseFile["realDataNpy"]
    testDataNpy = caseFile["testDataNpy"]
    accuList = list()
    # if "retrainSet" in caseFile:
    #    outputSet = join(outputPathOriginal, caseFile["retrainSet"])
    # else:
    #    outputSet = join(outputPathOriginal, str(retrainMode) + str(0) + ".npy")
    # dataSupply.loadIEETrainData(caseFile, outputSet)
    if "expIndex" not in caseFile:
        expIndex = str(int(np.random.randint(100, 100000)))
    else:
        expIndex = caseFile["expIndex"]
    expDir = join(modelsPath, str(retrainMode) + "_" + expIndex)
    if not exists(expDir):
        os.makedirs(expDir)
    errPath = join(caseFile["outputPathOriginal"], "errList.pt")
    if exists(errPath):
        errList = torch.load(errPath)
    else:
        improvePredict = ieepredict.IEEPredictor(caseFile["improveDataNpy"], caseFile["modelPath"], 0)
        dst2 = join(caseFile["outputPathOriginal"], "improveerror")
        improveDataSet, _ = improvePredict.load_data(caseFile["improveDataNpy"])
        _, errList = improvePredict.predict(improveDataSet, dst2, caseFile["improveDataPath"], True,
                                            caseFile["improveCSV"], 1, False)
        torch.save(errList, errPath)
    model = testModule.loadDNN(caseFile["modelPath"], "KPNet", None, False)
    model.eval()
    for exp in range(expNumber, expNumber2 + 1):
        DNN = loadDNN(caseFile, None)
        outputModel = join(modelsPath, str(retrainMode) + str(exp) + "_kpmodel.pt")
        outputLoss = join(modelsPath, str(retrainMode) + str(exp) + "_loss.npy")
        if "retrainSet" in caseFile:
            outputSet = join(outputPathOriginal, caseFile["retrainSet"])
        else:
            outputSet = join(outputPathOriginal, str(retrainMode) + str(exp) + ".npy")
        dataSupply.loadIEETrainData(caseFile, outputSet, errList)

        trainer = Trainer(outputSet, testDataNpy, realDataNpy, batchSize, False, 0, 0, 0
                          , False, True, True, 0, outputModel, outputLoss, DNN, Epochs)
        trainer.train()
        print("Using model:", outputModel)
        predictor = ieepredict.IEEPredictor(outputSet, outputModel, 0)
        trainDataSet, _ = predictor.load_data(outputSet)
        predictor2 = ieepredict.IEEPredictor(testDataNpy, outputModel, 0)
        testDataSet, _ = predictor.load_data(testDataNpy)
        outputTrainCSV = join(outputPathOriginal,
                              retrainMode + str(exp) + "_trainResult.csv")
        outputTestCSV = join(outputPathOriginal,
                             retrainMode + str(exp) + "_testResult.csv")
        dst = join(outputPathOriginal, "trainerror")
        dst2 = join(outputPathOriginal, "testerror")
        # counter = predictor.predict(trainDataSet, dst, None, True, outputTrainCSV, 1, False)
        predictor2.model = dnnModels.KPNet()
        if torch.cuda.is_available():
            weights = torch.load(outputModel, map_location=torch.device('cuda:0'))
        else:
            weights = torch.load(outputModel, map_location=torch.device('cpu'))
        predictor2.model.load_state_dict(weights.state_dict())

        counter2 = predictor2.predict(testDataSet, dst2, None, True, outputTestCSV, 0, False)
        imageList = pd.read_csv(outputTrainCSV)
        imageList2 = pd.read_csv(outputTestCSV)
        dictResult = {}
        cntTot = 0
        cntMC = 0
        for component in components:
            cntComp = 0
            cntTot = 0
            cntMC = 0
            for index, row in imageList.iterrows():
                cntTot += 1
                if row["result"] == "Wrong":
                    cntMC += 1
                    if row["worst_component"] == component:
                        cntComp += 1
            dictResult[component] = cntComp
        accuList.append(cntMC / cntTot)
        testAccuracy = 100.0 * ((1 - cntMC) / cntTot)
        if "retrainSet" not in caseFile:
            shutil.move(outputSet, join(expDir, str(retrainMode) + str(exp) + "_" +
                                        str(testAccuracy)[0:6] + ".npy"))
            shutil.move(outputModel, join(expDir, str(retrainMode) + str(exp) + "_" +
                                          str(testAccuracy)[0:6] + "_kpmodel.pt"))
            shutil.move(outputLoss, join(expDir, str(retrainMode) + str(exp) + "_" +
                                         str(testAccuracy)[0:6] + "_loss.npy"))
    return


def retrainAlex():
    global caseFile, eta, etaT, expIndex, BaggedUnsafeSet
    if caseFile["retrainMode"] != "SEDE":
        clusterUCs, totalAssigned, totalUc, totalUb, Ub = dataSupply.getUCs(caseFile, 0.3)  # U4/U5
        print("Total Assigned Images:", totalAssigned)
        print("US:", math.ceil(totalUc))
        print("BLUS:", math.ceil(totalUb))
        BaggedUnsafeSet = math.ceil(totalUb)
    retrainMode = caseFile["retrainMode"]
    if "retrieveAccuracy" in caseFile:
        retrieveAccuracy = caseFile["retrieveAccuracy"]
        newName = str(retrainMode) + "_" + str(caseFile["retrieveAccuracy"])
        DNNModels = caseFile["DNNsPath"] + newName
    else:
        retrieveAccuracy = None
        if "expIndex" not in caseFile:
            expIndex = str(int(np.random.randint(100, 100000)))
        else:
            expIndex = caseFile["expIndex"]
        DNNModels = join(caseFile["filesPath"], "DNNModels_" + retrainMode + "_" + expIndex)
    eta = etaT = "N/A"
    outputPath = caseFile["outputPath"]
    modelPath = caseFile["modelPath"]
    epochNum = caseFile["Epochs"]
    datasetName = caseFile["datasetName"]
    expNum = caseFile["expNum1"]
    expNum2 = caseFile["expNum2"]
    clsPath = caseFile["assignPTFile"]
    DataSets = join(outputPath, "DataSets")
    testSet = join(DataSets, "TestSet")
    if not exists(DNNModels):
        os.makedirs(DNNModels)
    imgClasses = caseFile["trainDataSet"].dataset.classes
    imageList = pd.read_csv(caseFile["testCSV"], names=["image", "result", "expected", "predicted"].append(imgClasses))
    cnt1 = 0
    resultDict = {}
    for index, row in imageList.iterrows():
        imagePath = row["image"]
        cnt1 += 1
        resultDict[imagePath] = {}
        resultDict[imagePath]["Old"] = row["result"]

    clsWithAssImages = torch.load(join(caseFile["filesPath"], "ClusterAnalysis_" + str(caseFile["clustMode"]),
                                       caseFile["selectedLayer"] + ".pt"))
    ##clsWithAssImages = torch.load(clsPath)
    clusterDistrib = list()
    clusterDict = {}
    for clusterID in clsWithAssImages['clusters']:
        clusterDict[clusterID] = {}
        clusterDict[clusterID]["Imp"] = clusterDict[clusterID]["Wor"] = 0
        if 'assigned' in clsWithAssImages['clusters'][clusterID]:
            clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
            if clustLen > 0:
                clusterDistrib.append(clustLen)
    print("UnsafeSet Distribution", clusterDistrib)
    print("Total:", sum(clusterDistrib))
    # print("Avg:", sum(clusterDistrib) / len(clusterDistrib))
    print("Total Clusters:", len(clsWithAssImages['clusters']))
    print("Assigned Clusters:", len(clusterDistrib))
    # print("UnsafeSet Size:", math.ceil(totalUb))
    # caseFile[retrainMode]["BaggedUnsafeSetSize"] = math.ceil(totalUb)
    print("RetrainMode:", retrainMode)
    ts = datasets.ImageFolder(root=caseFile["trainDataPath"] + "_R", transform=setupTransformer(datasetName))
    print("TrainingSet Size:", len(ts))
    caseFile[retrainMode]["failCount"] = caseFile[retrainMode]["totalFailCount"] = caseFile[retrainMode]["dupCount"] = 0
    # if retrainMode == "HUDD":
    #    bagPath = join(caseFile["filesPath"], "DataSets", retrainMode + "_" + str(expIndex) + "_toBag/")
    #    ts2, imagesList, caseFile = dataSupply.loadTrainData(bagPath, caseFile)  # UnsafeSet
    test = test2 = improvedList = worsenedList = improvedClustersList = list()
    expCounter = x = loadBar = 0
    dumbDict = {}
    numExps = expNum2 + 1 - expNum
    start = time.time()
    for exp in range(expNum, expNum2 + 1):

        clusterDistrib = list()
        clusterDict = {}
        for clusterID in clsWithAssImages['clusters']:
            clusterDict[clusterID] = {}
            clusterDict[clusterID]["Imp"] = clusterDict[clusterID]["Wor"] = 0
            if 'assigned' in clsWithAssImages['clusters'][clusterID]:
                clustLen = len(clsWithAssImages['clusters'][clusterID]['assigned'])
                if clustLen > 0:
                    clusterDistrib.append(clustLen)
        bestModelPath = join(DNNModels, retrainMode + "_" + str(exp) + "." +
                             str(basename(modelPath).split(".")[1]))
        if not (retrieveAccuracy is None):
            # DNN = loadDNN(caseFile, bestModelPath)
            for b in os.listdir(DNNModels):
                if b.startswith(join(retrainMode + "_" + str(exp))):
                    bestModelPath = join(DNNModels, b)
        else:
            DNN = loadDNN(caseFile, None)
            testAccuracy, resultDictNew = alexTest(bestModelPath, testSet, resultDict, datasetName, DNN, False, None)
            print("R:", testAccuracy.item())
            test.append(testAccuracy.item())
            tsList = list()
            randomID = int(np.random.randint(100, 100000))
            # randomID = 54207
            bagPath = join(caseFile["filesPath"], "DataSets",
                           retrainMode + "_" + str(randomID) + "_toBag/")
            print(bagPath)
            if not exists(bagPath):
                if "retrainSet" in caseFile:
                    dumbDict = torch.load(join(caseFile["DNNsPath"] + str(retrainMode) + "_" +
                                               str(caseFile["retrainSet"]).split("_")[1], caseFile["retrainMode"]
                                               + "_resultDict.pt"))
                    imagesList = dumbDict["UnsafeSet_" + str(caseFile["retrainSet"]).split("_")[0]]
                    ts2, imagesList, caseFile = dataSupply.loadTrainData(bagPath, caseFile, imagesList)
                else:
                    ts2, imagesList, caseFile = dataSupply.loadTrainData(bagPath, caseFile, None)  # UnsafeSet
                    dumbDict["UnsafeSet_" + retrainMode + str(exp)] = imagesList
            else:
                ts2 = datasets.ImageFolder(root=bagPath, transform=setupTransformer(datasetName))
            tsList.append(ts)
            tsList.append(ts2)

            ts3 = datasets.ImageFolder(root=caseFile["trainDataPath"] + "_S", transform=setupTransformer(datasetName))
            dataset_subset = torch.utils.data.Subset(ts3, np.random.choice(len(ts3), int(len(ts3) / 20), replace=False))
            tsList.append(dataset_subset)
            concatList = torch.utils.data.ConcatDataset(tsList)
            newTrainDataSet = torch.utils.data.DataLoader(concatList, batch_size=caseFile["batchSize"], shuffle=True,
                                                          num_workers=caseFile["workersCount"])
            _, DNN = alexTrain(caseFile, epochNum, newTrainDataSet, bestModelPath, DNN, None)
            # shutil.rmtree(bagPath)

        DNN = loadDNN(caseFile, bestModelPath)
        testAccuracy, resultDictNew = alexTest(bestModelPath, testSet, resultDict, datasetName, DNN, False, None)
        print("R:", testAccuracy.item())
        test.append(testAccuracy.item())
        improvedImages, worsenedImages, clusterDict = collectImprovedData(resultDictNew, clusterDict)
        testAccuracy, _ = alexTest(bestModelPath, testSet + "_S", None, datasetName, DNN, False, None)
        print("S:", testAccuracy.item())
        # bagPath = join(caseFile["filesPath"], "DataSets", retrainMode + "_" +
        #                       str(int(np.random.randint(100, 100000))) + "_toBag/")
        # dataSupply.generateTestSet(caseFile, bagPath)
        # tsList2 = list()
        # ts3 = PathImageFolder(root=bagPath, transform=setupTransformer(datasetName))
        # ts4 = PathImageFolder(root=caseFile["testDataPath"], transform=setupTransformer(datasetName))
        # tsList2.append(ts3)
        # tsList2.append(ts4)
        # concatList = torch.utils.data.ConcatDataset(tsList2)
        # newTestDataSet = torch.utils.data.DataLoader(concatList, batch_size=caseFile["batchSize"], shuffle=True,
        #                                              num_workers=caseFile["workersCount"])
        # testAccuracy2, _ = alexTest(bestModelPath, newTestDataSet, None, datasetName, DNN, False, None)
        # test2.append(testAccuracy2.item())
        # print(testAccuracy2.item())
        # shutil.rmtree(bagPath)
        improvedList.append(improvedImages)
        worsenedList.append(worsenedImages)
        expCounter += 1
        x += 1
        if int(x / (numExps * 0.1)) == 1:
            loadBar += 10.0
            spentTime = ((time.time() - start) / 60.0)
            timePerLoadBar = spentTime / loadBar
            spentTime = timePerLoadBar * loadBar
            fullTime = timePerLoadBar * 100
            remTime = math.ceil(fullTime - spentTime)
            if remTime > 60:
                etaT = str(remTime / 60)[0:4] + "hs."
            else:
                etaT = str(remTime) + " mins."
            x = 0
        eta = str(int(100.0 * exp / (numExps)))
        if retrieveAccuracy is None:
            shutil.move(join(DNNModels, retrainMode + "_" + str(exp) + "." +
                             str(basename(modelPath).split(".")[1])),
                        join(DNNModels, retrainMode + "_" + str(exp) + "_" + str(testAccuracy.item())[0:6] +
                             "." + str(basename(modelPath).split(".")[1])))
    print(test)
    print(sum(test) / len(test))
    print(test2)
    print(sum(test2) / len(test2))
    return sum(improvedList) / len(improvedList), max(improvedList), min(improvedList), \
           sum(worsenedList) / len(worsenedList), max(worsenedList), min(worsenedList), sum(test) / len(test), \
           DNNModels, dumbDict, clusterDict


def loadAlexRetrainDataSet():
    return


def updateCaseFile():
    return


def collectImprovedData(resultDictNew, clusterDict):
    global clsWithAssImages
    global caseFile
    clsWithAssImages = torch.load(caseFile["clsPath"])
    worsenedImages = 0
    improvedImages = 0
    for img in resultDictNew:
        if resultDictNew[img]["Old"] == "Correct":
            if resultDictNew[img]["New"] == "Wrong":
                worsenedImages += 1
        if resultDictNew[img]["Old"] == "Wrong":
            if resultDictNew[img]["New"] == "Correct":
                imgClass = basename(dirname(img))
                imgName = "Test_" + str(basename(img)).split(".")[0] + "_" + imgClass
                improvedImages += 1
                for clusterID in clsWithAssImages['clusters']:
                    if clsWithAssImages['clusters'][clusterID]['members'].count(imgName) > 0:
                        clusterDict[clusterID]["Imp"] += 1
    for clusterID in clusterDict:
        print(clusterID, clusterDict[clusterID]["Imp"])
    print("Worsened:", worsenedImages)
    return improvedImages, worsenedImages, clusterDict


def writeFile(textPath, input, text):
    file = open(textPath, "a")
    if text is not None:
        file.write(text + "\n")
    for i in input:
        file.write(str(i) + "\n")
    file.close()


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
        # net.eval()
        DNN = net
    else:
        if not scratchFlag:
            weights = torch.load(modelPath, map_location=torch.device('cpu'))
            if Alex:
                net.load_state_dict(weights)
            elif KP:
                net.load_state_dict(weights.state_dict())
        # net.eval()
        DNN = net
    return DNN


def alexTrain(caseFile, epochNum, trainData, bestModelPath, net, validationSet):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
    best_loss = 0
    x = 0
    ETA1 = 0
    ETA2 = 0

    x = 0
    trainAccuracy = list()
    print(len(trainData.dataset))
    torch.save(net.state_dict(), bestModelPath)
    for i in range(1, epochNum + 1):
        totalCounter = 0
        start1 = time.time()
        correct = 0
        net.train()
        # print(trainData)
        retrainLength = len(trainData)
        for b_idx, (data, classes) in enumerate(trainData):
            start1 = time.time()
            totalCounter += 1
            if torch.cuda.is_available():
                net.cuda()
                data, classes = data.cuda(), classes.cuda()
            else:
                data = data.cpu()
                classes = classes.cpu()
            data, classes = Variable(data), Variable(classes)
            optimizer.zero_grad()
            scores = net.forward(data)
            scores = scores.view(data.size()[0], caseFile["numClass"])
            _, prediction = torch.max(scores.data, 1)
            correct += torch.sum(prediction == classes.data).float()
            loss = criterion(scores, classes)
            loss.backward()
            optimizer.step()
            # if b_idx % log_schedule == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    i, (b_idx + 1) * len(data), len(trainData.dataset),
            #       100. * (b_idx + 1) * len(data) / len(trainData.dataset), loss.item()), end="\r")

            if x == 0:
                end = time.time()
                ETA1 = int((((end - start1) / 60.0) * retrainLength))
                ETA2 = int((ETA1 * epochNum))
                x += 1
            # if eta is not None:
            #    print("Checked:", str(totalCounter) + "/" + str(retrainLength) + " " +
            #          str(int(100.0 * totalCounter / retrainLength)) + "%",
            #          str(int(100.0 * i / epochNum)) + "%",
            #          eta + "%", "ETA:" + str(etaT), end="\r")
            # else:
            print("Checked:", str(totalCounter) + "/" + str(retrainLength) + " " +
                  str(int(100.0 * totalCounter / retrainLength)) + "%",
                  str(int(100.0 * i / epochNum)) + "%", end="\r")
        print(i)
        if validationSet is None:
            avg_loss = correct / len(trainData.dataset) * 100
        else:
            avg_loss, _ = alexTest(bestModelPath, validationSet, None, caseFile["datasetName"], net, False, None)
        print("loss:", avg_loss)
        if (avg_loss > best_loss):
            torch.save(net.state_dict(),
                       dirname(bestModelPath) + os.path.basename(bestModelPath).split(".")[0] + str(i) + ".pth")
            best_loss = avg_loss

            dataTransform = setupTransformer(caseFile["datasetName"])
            transformedData2 = PathImageFolder(root=caseFile["testDataPath"], transform=dataTransform)
            testData2 = torch.utils.data.DataLoader(transformedData2, batch_size=caseFile["batchSize"], shuffle=True,
                                                    num_workers=caseFile["workersCount"])

            testAccuracy2, resultDictNew = alexTest(bestModelPath, testData2, None, caseFile["datasetName"],
                                                    net, False, None)

            print(testAccuracy2.item())
        # print("training accuracy ({:.2f}%)".format(avg_loss))
        trainAccuracy.append(avg_loss)
    return trainAccuracy, net


def alexTrain_Original(caseFile, epochNum, trainData, bestModelPath, net, validationSet):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
    best_loss = 0
    x = 0
    ETA1 = 0
    ETA2 = 0
    bestNet = net
    best_t_loss = 0
    test_loss = 0
    x = 0
    trainAccuracy = list()
    print(len(trainData.dataset))
    torch.save(net.state_dict(), bestModelPath)
    for i in range(1, epochNum + 1):
        totalCounter = 0
        start1 = time.time()
        correct = 0
        bestNet.train()
        # print(trainData)
        retrainLength = len(trainData)
        for b_idx, (data, classes) in enumerate(trainData):
            start1 = time.time()
            totalCounter += 1
            if torch.cuda.is_available():
                bestNet.cuda()
                data, classes = data.cuda(), classes.cuda()
            else:
                data = data.cpu()
                classes = classes.cpu()
            data, classes = Variable(data), Variable(classes)
            optimizer.zero_grad()
            scores = bestNet.forward(data)
            scores = scores.view(data.size()[0], caseFile["numClass"])
            _, prediction = torch.max(scores.data, 1)
            correct += torch.sum(prediction == classes.data).float()
            loss = criterion(scores, classes)
            loss.backward()
            optimizer.step()
            # if b_idx % log_schedule == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    i, (b_idx + 1) * len(data), len(trainData.dataset),
            #       100. * (b_idx + 1) * len(data) / len(trainData.dataset), loss.item()), end="\r")

            if x == 0:
                end = time.time()
                ETA1 = int((((end - start1) / 60.0) * retrainLength))
                ETA2 = int((ETA1 * epochNum))
                x += 1
            # if eta is not None:
            #    print("Checked:", str(totalCounter) + "/" + str(retrainLength) + " " +
            #          str(int(100.0 * totalCounter / retrainLength)) + "%",
            #          str(int(100.0 * i / epochNum)) + "%",
            #          eta + "%", "ETA:" + str(etaT), end="\r")
            # else:
            print("Checked:", str(totalCounter) + "/" + str(retrainLength) + " " +
                  str(int(100.0 * totalCounter / retrainLength)) + "%",
                  str(int(100.0 * i / epochNum)) + "%", end="\r")
        train_loss = correct / len(trainData.dataset) * 100
        print("train loss:", train_loss)
        if (train_loss > best_loss):
            # if test_loss > best_t_loss:
            torch.save(net.state_dict(), bestModelPath)
            bestNet = net
            best_loss = train_loss
            print("model saved")

        if validationSet is not None:
            test_loss, _ = alexTest(bestModelPath, validationSet, None, caseFile["datasetName"], bestNet, False, None)
        # print("training accuracy ({:.2f}%)".format(avg_loss))
        trainAccuracy.append(train_loss)
        print("test loss:", test_loss)
    return trainAccuracy, net


def alexTest(bestModelPath, testSet, resultDict, datasetName, net, saveFlag, outPutFile):
    global best_accuracy
    global exp
    global epoch
    global caseFile
    correct = 0
    if exists(bestModelPath):
        if torch.cuda.is_available():
            weights = torch.load(bestModelPath)
        else:
            weights = torch.load(bestModelPath, map_location=torch.device('cpu'))
        net.load_state_dict(weights)
    net.eval()
    if resultDict is None:
        testData = testSet
    else:
        dataTransformer = setupTransformer(datasetName)
        transformedData = PathImageFolder(root=testSet, transform=dataTransformer)
        testData = torch.utils.data.DataLoader(transformedData, batch_size=caseFile["batchSize"], shuffle=True,
                                               num_workers=caseFile["workersCount"])
    classesStr = ','.join(str(class_) for class_ in testData.dataset.classes)
    if saveFlag:
        outFile = open(outPutFile, 'w')
        outFile.writelines("image,result,expected,predicted," + classesStr + "\r\n")
    totalInputs = 0
    for idx, (data, classes, paths) in enumerate(testData):
        if torch.cuda.is_available():
            data, classes = data.cuda(), classes.cuda()
        totalInputs += len(data)
        data, classes = Variable(data), Variable(classes)
        scores = net.forward(data)
        pred = scores.data.max(1)[1]
        correct += torch.sum(pred == classes.data).float()
        for i in range(len(data)):
            if (classes.data[i].eq(pred[i])):
                outcome = "Correct"
            else:
                outcome = "Wrong"
            # imageFileName = basename(paths[i])
            if resultDict is not None:
                resultDict[paths[i]]["New"] = outcome
            strExpected = testData.dataset.classes[classes[i]]
            strPred = testData.dataset.classes[pred[i].item()]
            scoreStr = ','.join([str(score) for score in scores[i].data.tolist()])
            if saveFlag:
                outFile.writelines(paths[i] + "," + outcome + "," + strExpected + "," + strPred + "," + scoreStr[1:len(
                    scoreStr) -
                                                                                                                   2] +
                                   "\r\n")
    print("Predicted {} out of {} correctly".format(correct, totalInputs))
    print("The average accuracy is: {} %".format(100.0 * correct / (float(totalInputs))))
    print("Total erronous" + str(totalInputs - correct))
    if saveFlag:
        outFile.close()
    # print("predicted {} out of {}".format(correct, len(testData.dataset)))
    val_accuracy = (correct / float(totalInputs)) * 100
    # print(val_accuracy)

    # now save the model if it has better accuracy than the best model seen so forward
    return val_accuracy, resultDict


def getSize(outputPath, bagSize, clsPath):
    labelDataSize = 0
    numClusters = 0
    bagSizeperCluster = bagSize
    clusterwithAssignedImages = torch.load(clsPath)
    for clusterID in clusterwithAssignedImages['clusters']:
        numClusters = numClusters + 1
        for image in clusterwithAssignedImages['clusters'][clusterID]['assigned']:
            labelDataSize = labelDataSize + 1
    toBag = (bagSizeperCluster * numClusters) - labelDataSize
    totalUnbaggedSize = labelDataSize
    totalBaggedSize = bagSizeperCluster * numClusters

    return numClusters, toBag, totalUnbaggedSize, totalBaggedSize


def testImage(model, inputs, labels, thresholdPixels, area):
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    predict = model(inputs)
    predict_cpu = predict.cpu()
    predict_cpu = predict_cpu.detach().numpy()
    errorList = testModule.ieeError(predict_cpu, labels, area, thresholdPixels)
    return errorList[0]


from imports import os, torch, datasets, transforms, Variable, nn, optim, setupTransformer
import dnnModels
import testModule

learning_rate = 0.001
momentum = 0.9
log_schedule = 10


def fineTune(modelPath, outputPath, datasetName, epochNum, caseFile):
    if datasetName == "FLD":
        DNN = loadDNN(caseFile, None)
        modelsPath = modelPath.split(".pt")[0]
        outputModel = join(modelsPath, "fineTuned_kpmodel.pt")
        outputLoss = join(modelsPath, "fineTuned_loss.npy")
        trainSet = join(outputPath, "newdata.npy")
        testSet = join(outputPath, "newtest.npy")
        if not isfile(trainSet):
            trainDataNpy = caseFile["trainDataNpy"]
            improveDataNpy = caseFile["realDataNpy"]
            retrainDataSet_X = []
            retrainDataSet_Y = []
            newtest_X = []
            newtest_Y = []
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
            for i in range(0, 1000):
                retrainDataSet_X.append(x_data[i])
                retrainDataSet_Y.append(y_data[i])
            for i in range(1000, 2000):
                newtest_X.append(x_data[i])
                newtest_Y.append(y_data[i])
            newtest_X1 = np.array(newtest_X)
            newtest_Y1 = np.array(newtest_Y)
            retrainDataSet_X1 = np.array(retrainDataSet_X)
            retrainDataSet_Y1 = np.array(retrainDataSet_Y)
            print("Size of RetrainSet:", str(len(retrainDataSet_X)))
            print("Size of TestSet:", str(len(newtest_X)))
            retrainDataSet = {"data": retrainDataSet_X1, "label": retrainDataSet_Y1}
            testDataSet = {"data": newtest_X1, "label": newtest_Y1}
            np.save(trainSet, retrainDataSet)
            np.save(testSet, testDataSet)
        trainer = Trainer(trainSet, testSet, caseFile["realDataNpy"], caseFile["batchSize"], False, 0, 0, 0
                          , False, True, True, 0, outputModel, outputLoss, DNN, epochNum)
        trainer.train()
        print("Using model:", outputModel)
        testKP(testSet, outputModel, outputPath)
    elif datasetName != "SAP":
        # outputPath = "/home/users/hfahmy/DEEP/HPC/FR"
        # outputPath = "/scratch/users/fpastore/DEEP/gazedetectionandanalysisdnn/Learning/HUDD/OD/"
        testData = join(outputPath, "DataSets", "TestSet")
        testData2 = join(outputPath, "DataSets", "TestSet_S")
        trainData = join(outputPath, "DataSets", "TrainingSet")
        validationData = join(outputPath, "DataSets", "ValidationSet")
        improvementData = join(outputPath, "DataSets", "ImprovementSet")
        _, unityData, _ = loadData(trainData, datasetName, 4, 2048, None, None)
        _, testData, _ = loadData(testData, datasetName, 4, 128, None, None)
        _, testData2, _ = loadData(testData2, datasetName, 4, 128, None, None)
        print(len(unityData.dataset.classes))
        net = loadDNNX(modelPath, "AlexNet", len(unityData.dataset.classes), scratchFlag=False)
        # print(unityData.dataset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
        best_loss = 0
        trainAccuracy = list()
        validAccuracy = list()
        # classes = torch.FloatTensor(unityData.dataset.classes)
        for i in range(1, epochNum + 1):
            print("Epoch", i)
            ##TRAIN
            correct = 0
            net = net.train()
            for b_idx, (data, classes, imgs) in enumerate(unityData):
                if torch.cuda.is_available():
                    net.cuda()
                    data, classes = data.cuda(), classes.cuda()
                # print(data.shape)
                data, classes = Variable(data), Variable(classes)
                optimizer.zero_grad()
                scores = net.forward(data)
                scores = scores.view(data.size()[0], len(unityData.dataset.classes))
                _, prediction = torch.max(scores.data, 1)
                correct += torch.sum(prediction == classes.data).float()
                loss = criterion(scores, classes)
                loss.backward()
                optimizer.step()
                if b_idx % log_schedule == 0:
                    print('Train Epoch: {} [{}\{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, (b_idx + 1) * len(data), len(unityData.dataset),
                           100. * (b_idx + 1) * len(data) / len(unityData.dataset), loss.item()), end='\r')
            avg_loss = correct / len(unityData.dataset) * 100
            print("training accuracy ({:.2f}%)".format(avg_loss))

            if (avg_loss > best_loss):
                net = net.eval()
                torch.save(net.state_dict(), join(outputPath, str(i) + "_finetunedModel.pth"))
                best_loss = avg_loss
                ##TEST

                model = loadDNNX(join(outputPath, str(i) + "_finetunedModel.pth"), "AlexNet",
                                 len(unityData.dataset.classes), scratchFlag=False)
                model = model.eval()
                testModule.testErrorAlexNet(model, testData, False, None)
                testModule.testErrorAlexNet(model, testData2, False, None)
        return net
    else:

        LOGDIR = join(outputPath, "DNNModels")
        sess = tf.compat.v1.Session()

        model = dnnModels.ConvModel()

        train_vars = tf.compat.v1.trainable_variables()
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * L2_REG
        train_step = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()

        saver.restore(sess, modelPath)
        layers = [model.hconv1, model.h_conv1, model.hconv2, model.h_conv2, model.hconv3, model.h_conv3,
                  model.hconv4,
                  model.h_conv4, model.hconv5, model.h_conv5]
        for layer in layers:
            layer.trainable = False

        print("loaded")
        min_loss = 1.0
        data_reader = dataSupply.DataReader(data_dir=join(outputPath, "DataSets"))
        start_step = 0
        # Train_img, Train_SA, Train_FID = data_reader.load_train_batch(data_reader.num_train_images)
        Test_img, Test_SA, Test_FID = data_reader.load_val_batch(data_reader.num_val_images)

        # train_path = os.path.join(LOGDIR, "traindata_RS.txt")
        test_path = os.path.join(LOGDIR, "testdata_RS.txt")
        # file1 = open(train_path, 'w')
        file2 = open(test_path, 'w')
        # for i in range(0, len(Train_FID)):
        #    file1.write('%s\n' % str(Train_FID[i]))
        #    file1.write('%s\n' % str(Train_SA[i]))
        for i in range(0, len(Test_FID)):
            file2.write('%s\n' % str(Test_FID[i]))
            file2.write('%s\n' % str(Test_SA[i]))

        tf.summary.scalar("loss", loss)
        # merged_summary_op = tf.compat.v1.summary.merge_all()
        # summary_writer = tf.summary.SummaryWriter(LOGDIR, graph=tf.compat.v1.get_default_graph())

        for i in range(start_step, start_step + NUM_STEPS):
            xs, ys, fid = data_reader.load_train_batch(BATCH_SIZE)
            sess.run(model.x, feed_dict={model.x: xs, model.y_: ys})
            sess.run(train_step, feed_dict={model.x: xs, model.y_: ys})
            train_error = sess.run(loss, feed_dict={model.x: xs, model.y_: ys})
            print("Step %d, train loss %g" % (i, train_error))

            if i % 100 == 0:

                val_xs, val_ys, val_fid = data_reader.load_val_batch(data_reader.num_val_images)
                val_error = loss.eval(session=sess, feed_dict={model.x: val_xs, model.y_: val_ys})
                # val_error = loss.eval(session=sess, feed_dict={model.x: Test_img, model.y_: Test_SA})

                sim_xs, sim_ys, sim_fid = data_reader.load_sim_batch(data_reader.num_sim_images)
                sim_error = loss.eval(session=sess, feed_dict={model.x: sim_xs, model.y_: sim_ys})

                print("val loss %g" % (val_error))
                print("sim loss %g" % (sim_error))
                print("Step %d, val loss %g" % (i, val_error))
                if i > 0 and i % 100 == 0:
                    if not os.path.exists(LOGDIR):
                        os.makedirs(LOGDIR)
                        checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, val_error))
                        filename = saver.save(sess, checkpoint_path)
                        print("Model saved in file: %s" % filename)
                    elif val_error < min_loss:
                        min_loss = val_error
                        if not os.path.exists(LOGDIR):
                            os.makedirs(LOGDIR)

                        checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, val_error))
                        filename = saver.save(sess, checkpoint_path)
                        print("Model saved in file: %s" % filename)

def testKP(testSet, outputModel, outputPath):
    testPredictor = ieepredict.IEEPredictor(testSet, outputModel, False, 0, 0)
    testDataSet, _ = testPredictor.load_data(testSet)
    outputTestCSV = join(outputPath, "fineTuned_testResult.csv")
    dst = join(outputPath, "testerror")
    testPredictor.model = dnnModels.KPNet()
    if torch.cuda.is_available():
        weights = torch.load(outputModel, map_location=torch.device('cuda:0'))
    else:
        weights = torch.load(outputModel, map_location=torch.device('cpu'))
    testPredictor.model.load_state_dict(weights.state_dict())
    testPredictor.predict(testDataSet, dst, None, True, outputTestCSV, 0, False, None)
    imageList = pd.read_csv(outputTestCSV)
    cntTot = 0
    cntMC = 0
    for index, row in imageList.iterrows():
        cntTot += 1
        if row["result"] == "Wrong":
            cntMC += 1
    testAccuracy = 100.0 * ((1 - cntMC) / cntTot)
    print("Avg.", testSet, "accuracy %", testAccuracy)
def genericTrain(outputPath, datasetName, epochNum):
    if datasetName != "SAP":
        # outputPath = "/home/users/hfahmy/DEEP/HPC/FR"
        # outputPath = "/scratch/users/fpastore/DEEP/gazedetectionandanalysisdnn/Learning/HUDD/OD/"
        testData = join(outputPath, "DataSets", "TestSet")
        trainData = join(outputPath, "DataSets", "TrainingSet")
        validationData = join(outputPath, "DataSets", "ValidationSet")
        improvementData = join(outputPath, "DataSets", "ImprovementSet")
        _, unityData, _ = loadData(trainData, datasetName, 4, 2056, None, None)
        _, testData, _ = loadData(testData, datasetName, 2, 512, None, None)
        print(len(unityData.dataset.classes))
        net = loadDNNX(None, "AlexNet", len(unityData.dataset.classes), scratchFlag=True)
        # print(unityData.dataset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
        best_loss = 0
        trainAccuracy = list()
        validAccuracy = list()
        # classes = torch.FloatTensor(unityData.dataset.classes)
        for i in range(1, epochNum + 1):
            print("Epoch", i)
            ##TRAIN
            correct = 0
            net = net.train()
            for b_idx, (data, classes, imgs) in enumerate(unityData):
                if torch.cuda.is_available():
                    net.cuda()
                    data, classes = data.cuda(), classes.cuda()
                # print(data.shape)
                data, classes = Variable(data), Variable(classes)
                optimizer.zero_grad()
                scores = net.forward(data)
                scores = scores.view(data.size()[0], len(unityData.dataset.classes))
                _, prediction = torch.max(scores.data, 1)
                correct += torch.sum(prediction == classes.data).float()
                loss = criterion(scores, classes)
                loss.backward()
                optimizer.step()
                if b_idx % log_schedule == 0:
                    print('Train Epoch: {} [{}\{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, (b_idx + 1) * len(data), len(unityData.dataset),
                           100. * (b_idx + 1) * len(data) / len(unityData.dataset), loss.item()), end='\r')
            avg_loss = correct / len(unityData.dataset) * 100
            print("training accuracy ({:.2f}%)".format(avg_loss))

            if (avg_loss > best_loss):
                net = net.eval()
                torch.save(net.state_dict(), join(outputPath, str(i) + "_pretrainedModel.pth"))
                best_loss = avg_loss
                ##TEST

                model = loadDNNX(join(outputPath, str(i) + "_pretrainedModel.pth"), "AlexNet",
                                 len(unityData.dataset.classes), scratchFlag=False)
                model = model.eval()
                testModule.testErrorAlexNet(model, testData, False, None)
    else:
        LOGDIR = join(outputPath, "DNNModels")
        sess = tf.compat.v1.Session()

        model = dnnModels.ConvModel()

        train_vars = tf.compat.v1.trainable_variables()
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * L2_REG
        train_step = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.compat.v1.global_variables_initializer())
        # sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        print("loaded")
        min_loss = 1.0
        data_reader = dataSupply.DataReader(data_dir=join(outputPath, "DataSets"))
        # data_reader_test = dataSupply.DataReader(data_dir=join(outputPath, "DataSets", "TrainingSet"))
        Train_img = []
        Train_FID = []
        Train_SA = []
        Test_img = []
        Test_FID = []
        Test_SA = []

        start_step = 0
        # start_step = float(args.restore_from.split('step-')[0].split('-')[-1])
        # print('Model restored from ' + args.logdir + '/' + args.restore_from)
        # Train_img, Train_SA, Train_FID = data_reader.load_train_batch(data_reader.num_train_images)

        # val_error = loss.eval(session=sess, feed_dict={model.x: Test_img, model.y_: Test_SA})
        # heatmaps = getActivations(model, Test_img[0:1], Test_SA[0:1])
        # im = Image.fromarray((np.array(Test_img[0] * 255).astype(np.uint8)))
        # im.save(os.path.join(args.logdir, "your_file.jpeg"))
        # hn = np.array(heatmaps[0][0])
        # hn = 255 * (hn - hn.min()) / (hn.max() - hn.min())
        # hn = hn.astype(np.uint8)
        # hn.reshape(200, 66)
        # print(hn)
        # im = Image.fromarray(hn)
        # im.save(os.path.join(args.logdir, "your_file2.jpeg"))
        # print(Test_img[0])
        # print(heatmaps[0][0])

        tf.summary.scalar("loss", loss)
        # merged_summary_op = tf.compat.v1.summary.merge_all()
        # summary_writer = tf.summary.SummaryWriter(LOGDIR, graph=tf.compat.v1.get_default_graph())

        for i in range(start_step, start_step + NUM_STEPS):
            xs, ys, fid = data_reader.load_train_batch(BATCH_SIZE)
            sess.run(model.x, feed_dict={model.x: xs, model.y_: ys})
            sess.run(train_step, feed_dict={model.x: xs, model.y_: ys})
            train_error = sess.run(loss, feed_dict={model.x: xs, model.y_: ys})
            print("Step %d, train loss %g" % (i, train_error))

            if i % 10 == 0:
                val_xs, val_ys, val_fid = data_reader.load_val_batch(BATCH_SIZE)
                val_error = loss.eval(session=sess, feed_dict={model.x: val_xs, model.y_: val_ys})
                # loss.eval(feed_dict={model.x: xs, model.y_: ys})
                print("Step %d, val loss %g" % (i, val_error))
                if i > 0 and i % 100 == 0:
                    if not os.path.exists(LOGDIR):
                        os.makedirs(LOGDIR)
                        checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, val_error))
                        filename = saver.save(sess, checkpoint_path)
                        print("Model saved in file: %s" % filename)
                    elif val_error < min_loss:
                        min_loss = val_error
                        if not os.path.exists(LOGDIR):
                            os.makedirs(LOGDIR)
                        checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, val_error))
                        filename = saver.save(sess, checkpoint_path)
                        print("Model saved in file: %s" % filename)
                        total_xs, total_sa, total_fid = data_reader.load_val_batch(data_reader.num_val_images)
                        val_error = loss.eval(session=sess, feed_dict={model.x: total_xs, model.y_: total_sa})
                        print("val loss %g" % (val_error))
        Train_img, Train_SA, Train_FID = data_reader.load_train_batch(data_reader.num_train_images)
        # train_error = loss.eval(session=sess, feed_dict={model.x: Train_img, model.y_: Train_SA})
        # Test_img, Test_SA, Test_FID = data_reader.load_val_batch(data_reader.num_val_images)
        Test_img, Test_SA, Test_FID = data_reader.load_val_batch(data_reader.num_val_images)

        train_path = os.path.join(LOGDIR, "traindata.txt")
        test_path = os.path.join(LOGDIR, "testdata.txt")
        file1 = open(train_path, 'w')
        file2 = open(test_path, 'w')
        for i in range(0, len(Train_FID)):
            file1.write('%s\n' % str(Train_FID[i]))
            file1.write('%s\n' % str(Train_SA[i]))
        for i in range(0, len(Test_FID)):
            file2.write('%s\n' % str(Test_FID[i]))
            file2.write('%s\n' % str(Test_SA[i]))

        def getActivations(model, xs, ys):
            layers = [model.hconv1, model.h_conv1, model.hconv2, model.h_conv2, model.hconv3, model.h_conv3,
                      model.hconv4,
                      model.h_conv4, model.hconv5, model.h_conv5, model.hfc1, model.h_fc1, model.hfc2, model.h_fc2,
                      model.hfc3, model.h_fc3, model.hfc4, model.h_fc4, model.y]
            activations = []
            print("Getting Activations..")
            for layer in layers:
                activations.append(sess.run(layer, feed_dict={model.x: xs, model.y_: ys}))
                print(activations[len(activations) - 1].shape)
            print("Getting Heatmaps..")
            model.relprob(sess, activations[-1], xs)


def loadDNNX(modelPath, modelArch: str, numClasses, scratchFlag):
    if modelArch == "AlexNet":
        net = dnnModels.AlexNetIEE(numClasses)  ### HPD
        # net = dnnModels.AlexNet(numClasses) ### GD - OC - ASL - TS - AC - OD
        if torch.cuda.is_available():
            if not scratchFlag:
                weights = torch.load(modelPath)
                net.load_state_dict(weights)
            net = net.to('cuda')
            net.cuda()
        else:
            if not scratchFlag:
                weights = torch.load(modelPath, map_location=torch.device('cpu'))
                net.load_state_dict(weights)
            net.eval()
    elif modelArch == "KPNet":
        print(modelArch)
        net = dnnModels.KPNet()
        if torch.cuda.is_available():
            if not scratchFlag:
                weights = torch.load(modelPath)
                net.load_state_dict(weights.state_dict())
            net = net.to('cuda')
            net.cuda()
        else:
            if not scratchFlag:
                weights = torch.load(modelPath, map_location=torch.device('cpu'))
                net.load_state_dict(weights.state_dict())
            net.eval()

    else:
        net = dnnModels.AlexNet(8)  # Default is GD
    return net


def loadData(dataPath: str, dataSetName: str, workersCount: int, batchSize: int, outputPath, weightPath):
    dataSet = 0
    train_di = 0
    imagesList = 0
    if dataSetName == "IEETEST":
        x = 0
        # ds = DataSupply.DataSupplier(using_gm=False)
        # if not isfile(outputPath):
        #    DataSupply.createData(dataPath, outputPath, weightPath)
        # train_di, valid_di, imagesList = ds.get_test_iter(outputPath)  # for test data
    # elif dataSetName == "IEETRAIN":
    #    ds = DataSupply.DataSupplier(using_gm=False)
    #    if not isfile(outputPath):
    #        DataSupply.createData(dataPath, outputPath)
    #    train_di = ds.get_train_iter(outputPath)  # for test data
    else:
        dataTransformer = setupTransformer(dataSetName)
        transformedData = PathImageFolder(root=dataPath, transform=dataTransformer)
        dataSet = torch.utils.data.DataLoader(transformedData, batch_size=batchSize, shuffle=True,
                                              num_workers=workersCount)
    return train_di, dataSet, imagesList


class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(PathImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class Trainer(object):
    def __init__(self, iee_train_data, iee_test_data, iee_real_data, batch_size, pin_memory, train_max_num,
                 test_max_num, real_max_num, multi_gpu, use_gpu_test, use_gpu_train, gpu_id, best_model_path,
                 loss_file_path, DNN, total_epoch):
        self.model = DNN
        self.iee_train_data = iee_train_data
        self.iee_test_data = iee_test_data
        self.iee_real_data = iee_real_data
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.train_max_num = train_max_num
        self.test_max_num = test_max_num
        self.real_max_num = real_max_num
        self.multi_gpu = multi_gpu
        self.use_gpu_test = use_gpu_test
        self.gpu_id = gpu_id
        self.use_gpu_train = use_gpu_train
        self.best_model_path = best_model_path
        self.loss_file_path = loss_file_path
        self.total_epoch = total_epoch
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print("switch to DataParallel mode")
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-3)
        self.lowest_mse = np.inf

    def get_train_data(self):
        data_supplier = DataSupplier(self.iee_train_data, self.batch_size, True, self.pin_memory, self.train_max_num)
        return data_supplier.get_data_iters()

    def get_test_data(self):
        data_supplier = DataSupplier(self.iee_test_data, self.batch_size, True, self.pin_memory, self.test_max_num)
        return data_supplier.get_data_iters()

    def get_real_data(self):
        data_supplier = DataSupplier(self.iee_real_data, self.batch_size, True, self.pin_memory, self.real_max_num)
        return data_supplier.get_data_iters()

    def loss_fn(self, predict, label):
        loss = torch.nn.MSELoss()
        # if weight:
        #    predict = predict*weight
        return loss(predict, label)

    def validate(self, data_batches):
        with torch.no_grad():
            loss_valid = []
            for idx, (inputs, labels) in enumerate(data_batches):
                labels = labels["gm"]
                if self.use_gpu_test:
                    if not self.multi_gpu:
                        self.model = self.model.cuda(self.gpu_id)
                        inputs, labels = Variable(inputs.cuda(self.gpu_id)), Variable(labels.cuda(self.gpu_id))
                    else:
                        self.model = self.model.cuda()
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                predict = self.model(inputs.float())
                loss = self.loss_fn(predict, labels.float())
                loss_valid.append(loss.data.item())

        return loss_valid

    def save_best(self, train_loss, test_loss, real_loss):
        # v_mse = np.mean(test_loss)
        v_mse = np.mean(train_loss)
        if v_mse < self.lowest_mse:
            self.lowest_mse = v_mse
            torch.save(self.model, self.best_model_path)
            loss = {"train": train_loss, "test": test_loss, "real": real_loss}
            np.save(self.loss_file_path, loss)
            # print("INFO: save model to: ", self.best_model_path, "save loss data to: ", self.loss_file_path)

    def train(self):
        print("INFO: loading training data...")
        train_data = self.get_train_data()
        print("INFO: loading test data...")
        test_data = self.get_test_data()
        print("INFO: loading kaggle data...")
        real_data = self.get_real_data()
        print("INFO: starting training ...")
        start = time.time()
        ETA1 = 0
        ETA2 = 0
        x = 0
        for epoch in range(self.total_epoch):
            loss_train = []
            totalCounter = 0
            retrainLength = len(train_data)
            t_s = time.time()
            for idx, (inputs, labels) in enumerate(train_data):
                start1 = time.time()
                totalCounter += 1
                labels = labels["gm"]
                # print("input shape: ", inputs.shape, "labels shape: ", labels.shape)
                if self.use_gpu_train:
                    if not self.multi_gpu:
                        self.model = self.model.cuda(self.gpu_id)
                        inputs, labels = Variable(inputs.cuda(self.gpu_id)), Variable(labels.cuda(self.gpu_id))
                    else:
                        self.model = self.model.cuda()
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                predict = self.model(inputs.float())
                loss = self.loss_fn(predict, labels.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_train.append(loss.data.item())
                if x == 0:
                    end = time.time()
                    ETA1 = int((((end - start1) / 60.0) * retrainLength))
                    ETA2 = int((ETA1 * self.total_epoch))
                    x += 1
                print("Checked:", str(totalCounter) + "/" + str(retrainLength) + " " +
                      str(int(100.0 * totalCounter / retrainLength)) + "%", " ETA1: <" +
                      str(ETA1 - math.ceil(ETA1 * (totalCounter / retrainLength))) + "m. " + " ETA2: <" +
                      str(math.ceil((ETA2 - int(ETA2 * (epoch / self.total_epoch))) / 60.0)) + "h. " +
                      str(int(100.0 * epoch / self.total_epoch)) + "%", end="\r")

            loss_test = self.validate(test_data)
            loss_real = self.validate(real_data)

            t_e = time.time()
            # print('\nEpoch: {}  Loss Train: {}, Loss Test: {},  Loss Real: {}, time: {} seconds '.format(epoch, np.mean(loss_train),
            #    np.mean(loss_test), np.mean(loss_real), int(t_e-t_s)), end='\n')
            self.save_best(loss_train, loss_test, loss_real)


def reportAvgImages(testDataPath, retrainMode, counter1, counter2, outputPath, datasetName, modelExtension, modelArch,
                    numClass, unimprovedList):
    modelPath = join(outputPath, "DNNModels")
    worsened = 0
    improved = 0
    maxWorsened = 0
    minWorsened = 50000000

    maxImproved = 0
    minImproved = 50000000

    # unimprovImagesX = open(unimprovedList, "r")
    # unimprovedImages = list()
    # for line in unimprovImagesX:
    #    unimprovedImages.append(line.split("\n")[0])
    # for line in f:
    # print(line)
    # print("Total images to skip", str(len(unimprovedImages)))
    ieeData, unityData, imgList = testModule.loadData(testDataPath, datasetName, 4, 128, None, None)
    testResultPath = join(outputPath, "testResult.csv")
    imgClasses = unityData.dataset.classes
    imageList = pd.read_csv(testResultPath,
                            names=["image", "result", "expected", "predicted"].append(imgClasses))
    cnt1 = 0
    resultDict = {}
    accuList = list()

    expirement = int(np.random.randint(100, 100000))
    textReport = join(modelPath, "Report_" + retrainMode + "_" + str(expirement) + ".txt")
    for index, row in imageList.iterrows():
        imagePath = row["image"]
        imageFileName = basename(imagePath)
        # if unimprovedImages.count(imageFileName)<1:
        cnt1 = cnt1 + 1
        if cnt1 % 100 == 0:
            print("Image checked: " + str(cnt1) + "/" + str(len(imageList)), end="\r")
        resultDict[imageFileName] = {}
        resultDict[imageFileName]["Old"] = row["result"]

    for i in range(counter1, counter2 + 1):
        file = open(textReport, "a")
        print(join(modelPath, retrainMode + "_" + str(i) + "." + modelExtension))
        w, i, accu = reportImages(datasetName, join(modelPath, retrainMode + "_" + str(i) + "." + modelExtension),
                                  resultDict, testDataPath, modelArch, numClass, None)
        accuList.append(accu)
        file.write(str(accuList))
        file.close()
        worsened += w
        improved += i
        if (w > maxWorsened):
            maxWorsened = w
        if (w < minWorsened):
            minWorsened = w
        if (i > maxImproved):
            maxImproved = i
        if (i < minImproved):
            minImproved = i
    total = counter2 - counter1 + 1
    print("Avg worsened: ", str((worsened / total)))
    print("Max worsened: ", str((maxWorsened)))
    print("Min worsened: ", str((minWorsened)))
    print("Avg improved: ", str((improved / total)))
    print("Max improved: ", str((maxImproved)))
    print("Min improved: ", str((minImproved)))
    print("Total Images: ", str(cnt1))
    print("Avg: ", str(sum(accuList) / len(accuList)))
    print(accuList)


def reportImages(datasetNameX, modelPath, resultDict, testDataPath, modelArch, numClass, improvImages):
    ieeData, unityData, imgList = testModule.loadData(testDataPath, datasetNameX, 4, 128, None, None)
    dnn = testModule.loadDNN(modelPath, modelArch, numClass, False)
    dnn = dnn.eval()
    totalInputs = 0
    loopIndex = 0
    correct = 0
    for idx, (batch, classes, paths) in enumerate(unityData):
        # print("loop " + str(loopIndex))
        # print("tested inputs " + str(totalInputs))
        loopIndex = loopIndex + 1
        batch, classes = Variable(batch), Variable(classes)
        if torch.cuda.is_available():
            batch, classes, dnn = batch.cuda(), classes.cuda(), dnn.cuda()
        scores = dnn(batch)
        scores = scores.detach()
        pred = scores.data.max(1)[1]
        for i in range(len(batch)):
            imageFileName = basename(paths[i])
            # if(improvImages.count(imageFileName) < 1):
            if (classes.data[i].eq(pred[i])):
                outcome = "Correct"
                correct += 1
            else:
                outcome = "Wrong"
            resultDict[imageFileName]["New"] = outcome
            totalInputs += 1

    worsenedImages = 0
    improvedImages = 0
    for img in resultDict:
        if resultDict[img]["Old"] == "Correct":
            if resultDict[img]["New"] == "Wrong":
                worsenedImages += 1
        if resultDict[img]["Old"] == "Wrong":
            if resultDict[img]["New"] == "Correct":
                improvedImages += 1

    print("Worsened", str(worsenedImages))
    print("Improved", str(improvedImages))
    print("Total images", str(totalInputs))
    print("Accuracy", str(correct / totalInputs))
    return worsenedImages, improvedImages, ((correct / totalInputs) * 100.00)


## load and verify model classes
def loadModel(modelPath: str, modelArch: str, classNum: int):
    if modelArch == "AlexNet":
        net = alexNet.AlexNet(classNum)
        if torch.cuda.is_available():
            weights = torch.load(modelPath)
        else:
            weights = torch.load(modelPath, map_location='cpu')
        net.load_state_dict(weights)
        net.eval()
        # if net.features
        return net
    else:
        return None


##### utility functions and classes
class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(PathImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


## load data
# def loadData(dataPath: str, workersCount: int, batchSize: int):
#    transformedData = PathImageFolder(root=dataPath, transform=None)
#    dataSet = torch.utils.data.DataLoader(transformedData, batch_size=batchSize, shuffle=True, num_workers=workersCount)
#    return dataSet


def calcImprovment(orgModel, desModel, testSetPath: str, dataSetName: str):
    testDataset = loadData(dataPath=testSetPath, workersCount=5, batchSize=32)
    transformer = setupTransformer(dataSetName)
    fixed = 0
    bugged = 0
    allBugsByOrg = 0
    allCorrectsByOrg = 0
    allBugsByDes = 0
    allCorrectsByDes = 0
    # cnt=0
    # allImagesCnt=len(testDataset.dataset.imgs)
    for img in testDataset.dataset.imgs:
        # cnt=cnt+1
        # if cnt % 100==0:
        #	print("checked: " + str(cnt) + "/"+ str(allImagesCnt))
        image = Image.open(img[0])
        imageTensor = transformer(image).float()
        imageTensor = imageTensor.unsqueeze_(0)
        imageTensor = Variable(imageTensor, requires_grad=False)
        imageTensor.detach()
        isCorrectByOrg = testModelForImg(orgModel, imageTensor, img[1])
        isCorrectByDes = testModelForImg(desModel, imageTensor, img[1])
        if not isCorrectByOrg:
            allBugsByOrg = allBugsByOrg + 1
        else:
            allCorrectsByOrg = allCorrectsByOrg + 1
        if not isCorrectByDes:
            allBugsByDes = allBugsByDes + 1
        else:
            allCorrectsByDes = allCorrectsByDes + 1
        if not isCorrectByOrg and isCorrectByDes:
            fixed = fixed + 1
        elif isCorrectByOrg and not isCorrectByDes:
            bugged = bugged + 1
    return fixed, bugged, fixed / allBugsByOrg, bugged / allCorrectsByOrg, allCorrectsByOrg, allBugsByOrg, allCorrectsByDes, allBugsByDes


def testModelForImg(model, image: torch.tensor, expectedClassID):
    scores = model(image)
    scores = scores.detach()
    pred = scores.data.max(1)[1].item()
    if (expectedClassID == pred):
        return True
    else:
        return False


###############

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    # parser.add_argument('--traningData',help='tranind data folder',required=False)

    parser.add_argument('-t', '--testSetPath', help='Train set folder', required=True)
    parser.add_argument('-m', '--orgModel', help='', required=True)
    parser.add_argument('-d', '--desModel', help='', required=True)
    parser.add_argument('-N', '--dataSetName', help='name of the dataset', required=True)
    parser.add_argument('-c', '--classNum', help='number of classes', required=True)
    parser.add_argument('-r', '--range', help='ranges for des model name', required=False)
    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)

    args = parser.parse_args()
    models = []

    if not args.range == None:
        startRange = int(args.range.split(',')[0])
        endRange = int(args.range.split(',')[1])
        for i in range(startRange, endRange + 1):
            models.append(args.desModel + str(i) + ".pth")
    else:
        models.append(args.desModel)
    print("Loading model  from " + args.orgModel)
    orgModel = loadModel(args.orgModel.strip(), "AlexNet", int(args.classNum))
    for model in models:
        print("testing model from " + model)
        desModel = loadModel(model.strip(), "AlexNet", int(args.classNum))
        print(calcImprovment(orgModel, desModel, args.testSetPath.strip(), args.dataSetName.strip()))

# if __name__ == '__main__':
#    trainer = Trainer()
#    trainer.train()
