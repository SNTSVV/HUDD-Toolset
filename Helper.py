#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#
from __future__ import print_function

import RQ1, RQ2
from imports import PathImageFolder, torch, os, argparse, setupTransformer, np, Variable, cv2, pd, random, shutil, \
    itemgetter, math, stat, datasets, imageio, join, isfile, exists, basename, Image, tqdm, hashlib, makedirs, tf
import testModule, dnnModels, HeatmapModule, clusterModule, assignModule, retrainModule, ieepredict, paramsModule, dataSupplier

components = ["noseridge", "nose", "mouth", "rightbrow", "righteye", "lefteye"]

class Helper(object):
    def KPNet(self, faceSubset):  # IEE
        self.saveResult()
        self.generateHeatmaps()
        self.generateHMDistances()
        self.generateClusters()
        self.selectLayer()
        self.generateConcepts()
        if self.RQ1A:
            self.updateCaseFile()
            RQ1.IEERQ1(self.caseFile)
            return
        self.simParam = False
        if self.simParam:
            self.updateCaseFile()
            self.generateImages()
        else:
            self.assignImages()
        return self.ResultDict, self.assignMode

    def AlexNet(self):  # GD - OC - ASL - TS - AC - HPD - OD
        self.saveResult()
        self.generateHeatmaps()
        self.generateHMDistances()
        self.generateClusters()
        self.selectLayer()
        if self.RQ1A:
            if self.datasetName.startswith("HPD"):
                RQ1.IEERQ1(self.caseFile)
            else:
                RQ1.UnityRQ1(self.caseFile)
            return
        self.simParam = False
        if self.simParam:
            self.generateImages()
        else:
            self.assignImages()
        return self.ResultDict, self.assignMode

    def __init__(self, outputPath, modelName, workersCount, batchSize, metric, clustFlag, assignFlag, retrainFlag,
                 retrainMode, retrainApproach, expNumber, expNumber2, bagSize, clustMode, assMode,
                 overWrite, selectionMode, FLD, cleanFlag, RCC, scratchFlag, retrieveAccuracy, RQ1A, retrainSet,
                 drawClustFlag, ieeVersion, clustNum):
        self.ResultDict = {}
        self.clustNum = int(clustNum) if (clustNum is not None) else 1
        datasetName = basename(outputPath)
        if isfile(join(outputPath, "caseFile.pt")):
            self.caseFile = torch.load(join(outputPath, "caseFile.pt"))
        else:
            self.caseFile = {}
        if RCC == "TT":
            self.saveHMTrainFlag = True
            self.saveHMTestFlag = True
            self.RCC = RCC
        else:
            self.saveHMTrainFlag = False
            self.saveHMTestFlag = True
            self.RCC = "T"
        if ieeVersion:
            self.iee_version = ieeVersion
            print("Using IEE Simulator V", self.iee_version)
        self.calcFlag = False
        self.faceSubset = "None_RCC"
        self.trainDataNpy = None
        self.testDataNpy = None
        self.improveDataNpy = None
        self.outputPath = outputPath
        self.outputPathOriginal = self.outputPath
        self.DataSetsPath = join(self.outputPath, "DataSets")
        self.trainDataPath = join(self.DataSetsPath, "TrainingSet")
        self.testDataPath = join(self.DataSetsPath, "TestSet")
        self.improveDataPath = join(self.DataSetsPath, "ImprovementSet", "ImprovementSet")
        self.realDataPath = join(self.DataSetsPath, "ImprovementSet", "ImprovementSet_Real")
        self.realDataNpy = None
        self.trainCSV = join(self.outputPath, "trainResult.csv")
        self.testCSV = join(self.outputPath, "testResult.csv")
        self.improveCSV = join(self.outputPath, "improveResult.csv")
        self.selectedLayer = None
        self.maxClust = 150
        self.batchSize = batchSize if (batchSize is not None) else 128
        self.workersCount = workersCount if (workersCount is not None) else 4
        if datasetName == "FLD":
            self.modelName = modelName if (modelName is not None) else "kpmodel.pt"
            self.numClass = 0
            self.simParam = True
            self.modelArch = "KPNet"
            self.Alex = False
            self.KP = True
            self.CN = False
            self.layers = ['Layer0', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'Layer7', 'Layer8',
                           'Layer9']
            self.datasetName = datasetName
            self.FLD = 2 if (FLD is None) else FLD
            self.imgExt = ".png"
            self.modelPath = join(self.outputPath, "DNNModels", self.modelName)
            net = dnnModels.KPNet()
            #self.scratchFlag = False
            #self.loadDNN(net)
            #fout = open(join(self.outputPath, "DNNModels", "kpmodel_pytorch.pt"), 'w')
            #for k, v in self.DNN.state_dict().items():
            #    fout.write(str(k) + '\n')
            #    fout.write(str(v.tolist()) + '\n')
            #fout.close()
            #exit(0)
            self.trainDataNpy = join(self.outputPath, "IEEPackage", "ieetrain.npy")
            self.testDataNpy = join(self.outputPath, "IEEPackage", "ieetest.npy")
            self.improveDataNpy = join(self.outputPath, "IEEPackage", "ieeimprove.npy")
            self.realDataNpy = join(self.outputPath, "IEEPackage", "ieereal.npy")
            self.trainPredict = ieepredict.IEEPredictor(self.trainDataNpy, self.modelPath, False, 0, 0)
            self.trainDataSet, _ = self.trainPredict.load_data(self.trainDataNpy)
            self.testPredict = ieepredict.IEEPredictor(self.testDataNpy, self.modelPath, False, 0, 0)
            self.testDataSet, _ = self.testPredict.load_data(self.testDataNpy)
            if not exists(self.testDataPath):
                self.testPredict.predict(self.testDataSet, self.testDataPath, self.testDataPath, True, self.testCSV, 0, True, None)

            ieepredict.ensure_folder(self.trainDataPath)
            ieepredict.ensure_folder(self.testDataPath)
            ieepredict.ensure_folder(self.improveDataPath)
            self.improvePredict = ieepredict.IEEPredictor(self.improveDataNpy, self.modelPath, False, 0, 0)
            self.improveDataSet, _ = self.improvePredict.load_data(self.improveDataNpy)
            self.realPredict = ieepredict.IEEPredictor(self.realDataNpy, self.modelPath, False, 0, 0)
            self.realDataSet, _ = self.realPredict.load_data(self.realDataNpy)
            self.Epochs = 50
        elif datasetName == "SAP":
            if modelName is not None:
                self.modelName = modelName
            else:
                self.modelName = "model-step-2900-val-0.0718435.ckpt"
            self.numClass = 1
            self.simParam = False
            self.modelArch = "ConvNet"
            self.Alex = False
            self.KP = False
            self.CN = True
            self.layers = ['Layer0', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'Layer7', 'Layer8',
                           'Layer9']
            self.datasetName = datasetName
            self.imgExt = ".png"
            self.modelPath = join(self.outputPath, "DNNModels", self.modelName)
            net = dnnModels.ConvModel()
            #data_reader_train = dataSupplier.DataReader(data_dir=join(outputPath, "DataSets", "TrainingSet"))
            #data_reader_test = dataSupplier.DataReader(data_dir=join(outputPath, "DataSets", "TrainingSet"))
            #self.trainDataSet, Train_SA, Train_FID = data_reader_train.load_all()
            #self.testDataSet, Test_SA, Test_FID = data_reader_test.load_all()
            self.Epochs = 1e5
        else:
            self.modelArch = "AlexNet"
            self.Alex = True
            self.KP = False
            self.CN = False
            self.FLD = 0
            self.datasetName = datasetName
            print(datasetName)
            if datasetName == "GD":
                self.simParam = True
                self.numClass = 8
                self.Epochs = 10
                self.imgExt = ".jpg"
                self.modelName = modelName if (modelName is not None) else "pretrainedModel.pth"
                net = dnnModels.AlexNet(self.numClass)
            elif datasetName == "OC":
                self.simParam = True
                self.numClass = 2
                self.Epochs = 10
                self.imgExt = ".jpg"
                self.modelName = modelName if (modelName is not None) else "pretrainedModel.pth"
                net = dnnModels.AlexNet(self.numClass)
            elif datasetName == "ASL":
                self.simParam = True
                self.numClass = 29
                self.Epochs = 13
                self.imgExt = ".jpg"
                self.modelName = modelName if (modelName is not None) else "pretrainedModel.pth"
                net = dnnModels.AlexNet(self.numClass)
            elif datasetName == "TS":
                self.simParam = False
                self.numClass = 43
                self.Epochs = 12
                self.imgExt = ".ppm"
                self.modelName = modelName if (modelName is not None) else "pretrainedModel.pty"
                net = dnnModels.AlexNet(self.numClass)
            elif datasetName == "OD":
                self.simParam = False
                self.numClass = 2
                self.Epochs = 13
                self.imgExt = ".jpg"
                self.modelName = modelName if (modelName is not None) else "13_pretrainedModel.pth"
                net = dnnModels.AlexNet(self.numClass)
            elif datasetName == "AC":
                self.simParam = False
                self.numClass = 8
                self.Epochs = 20
                self.imgExt = ".jpg"
                self.modelName = modelName if (modelName is not None) else "pretrainedModel.pth"
                net = dnnModels.AlexNet(self.numClass)
                # genericTrain.train(self.outputPath, self.datasetName, self.Epochs)
                # return
            elif datasetName == "HPD":
                self.simParam = True
                self.numClass = 9
                self.Epochs = 13
                #self.Epochs = 18 #HPD1
                #self.Epochs = 25 #HPD2
                #self.Epochs = 17
                self.datasetName = datasetName
                self.modelName = modelName if (modelName is not None) else "25_pretrainedModel.pth"
                self.modelName_S = "pretrainedModel.pth" #HPD-TR
                self.modelName = "pretrainedModel.pth" #HPD-TR
                self.modelName_R = "pretrainedModel.pth" #HPD-TR
                #self.modelName_R = "9_finetunedModel.pth" #HPD1 #"18_pretrainedModel.pth"
                net = dnnModels.AlexNetIEE(self.numClass)
                self.imgExt = ".png"
                self.testDataNpy = join(self.outputPath, "DataSets", "TestSet.npy")
                self.trainDataNpy = join(self.outputPath, "DataSets", "TrainingSet.npy")
                self.improveDataNpy = join(self.outputPath, "DataSets", "ImprovementSet.npy")
                self.realDataNpy = join(self.outputPath, "DataSets", "ieereal.npy")
                self.modelPath = join(self.outputPath, "DNNModels", self.modelName)
                self.trainPredict = ieepredict.IEEPredictor(self.trainDataNpy, self.modelPath, True, 9, 0)
                self.trainDataSet, _ = self.trainPredict.load_data(self.trainDataNpy)
                self.testPredict = ieepredict.IEEPredictor(self.testDataNpy, self.modelPath, True, 9, 0)
                self.testDataSet, _ = self.testPredict.load_data(self.testDataNpy)
                #self.testPredict.predict(self.testDataSet, dst, originalDst, saveFlag, saveImgs, mainCounter)
                self.improvePredict = ieepredict.IEEPredictor(self.improveDataNpy, self.modelPath, True, 9, 0)
                self.improveDataSet, _ = self.improvePredict.load_data(self.improveDataNpy)
                self.realPredict = ieepredict.IEEPredictor(self.realDataNpy, self.modelPath, True, 9, 0)
                self.realDataSet, _ = self.realPredict.load_data(self.realDataNpy)
                # genericTrain.train(self.outputPath, self.datasetName, self.Epochs)
                # return
            self.modelPath = join(self.outputPath, "DNNModels", self.modelName)
            self.layers = ['Layer0', 'Layer1', 'Layer3', 'Layer4', 'Layer6', 'Layer7', 'Layer9', 'Layer11', 'Layer13',
                           'Layer15', 'Layer18']
            dataTransformer = setupTransformer(self.datasetName)
            transformedData = PathImageFolder(root=self.trainDataPath, transform=dataTransformer)
            self.trainDataSet = torch.utils.data.DataLoader(transformedData, batch_size=self.batchSize, shuffle=True,
                                                            num_workers=self.workersCount)
            transformedData = PathImageFolder(root=self.testDataPath, transform=dataTransformer)
            self.testDataSet = torch.utils.data.DataLoader(transformedData, batch_size=self.batchSize, shuffle=True,
                                                           num_workers=self.workersCount)
            transformedData = PathImageFolder(root=join(self.improveDataPath),
                                              transform=dataTransformer)
            self.improveDataSet = torch.utils.data.DataLoader(transformedData, batch_size=self.batchSize, shuffle=True,
                                                              num_workers=self.workersCount)

        self.scratchFlag = scratchFlag if (scratchFlag is not None) else False
        self.loadDNN(net)
        #params = net.state_dict()
        #print(net.state_dict)
        #for key in net.features.parameters():
        #    key.requires_grad = False ##Freeze
        #print(params.keys())
        self.saveHMFlag = True
        self.computeFlag = True

        self.metric = metric if (metric is not None) else "Euc" #"Man"
        self.clustFlag = clustFlag if (clustFlag is not None) else True
        self.drawClustFlag = drawClustFlag if (drawClustFlag is not None) else True
        self.RQ1A = RQ1A if (RQ1A is not None) else False
        self.retrainFlag = retrainFlag if (retrainFlag is not None) else True
        self.retrainMode = retrainMode if (retrainMode is not None) else "None"
        self.overWrite = overWrite if (overWrite is not None) else False
        self.retrainApproach = retrainApproach if (retrainApproach is not None) else "A"
        self.expNumber = int(expNumber) if (expNumber is not None) else 1
        self.expNumber2 = int(expNumber2) if (expNumber2 is not None) else 10
        self.bagSize = int(bagSize) if (bagSize is not None) else 0
        self.selectionMode = selectionMode if (selectionMode is not None) else "WICD"
        self.clustMode = clustMode if (clustMode is not None) else "WICDWard"
        self.assignMode = assMode if (assMode is not None) else "ClosestU" #Entropy
        self.assignFlag = assignFlag if (assignFlag is not None) else True
        self.cleanFlag = cleanFlag if (cleanFlag is not None) else True
        self.saveTrainFlag = False if (exists(self.trainCSV)) else True
        self.saveTestFlag = False if (exists(self.testCSV)) else True
        self.saveImproveFlag = False if (exists(self.improveCSV)) else True
        #self.saveImproveFlag = False
        if self.assignFlag:
            assignPath = join(self.outputPath, "ClusterAnalysis_" + str(self.clustMode), "Assignments",
                              self.assignMode, self.selectionMode, "clusterwithAssignedImages.pt")
            if self.overWrite:
                if exists(assignPath):
                    shutil.rmtree(join(self.outputPath, "ClusterAnalysis_" + str(self.clustMode),
                                       "Assignments", self.assignMode, self.selectionMode))
            if exists(assignPath):
                self.assignFlag = False
        self.retrieveAccuracy = retrieveAccuracy
        self.retrainSet = retrainSet
        self.caseFile["retrainList"] = []
        #print(self.improveDataPath)
        for src_dir, dirs, files in os.walk(self.improveDataPath):
            for file_ in files:
                #print(file_)
                if (file_.endswith(".jpg")) or (file_.endswith(".png")) or (file_.endswith(".ppm")):
                    self.caseFile["retrainList"].append(join(src_dir, file_))
        self.updateCaseFile()
        print("Case Study Initialization Completed..")

    def cleanDirectories(self):
        setsPath = join(str(self.caseFile["filesPath"]), "DataSets")
        if not exists(setsPath):
            os.makedirs(setsPath)
        setsList = os.listdir(setsPath)
        for set in setsList:
            if set.startswith(self.retrainMode):
                shutil.rmtree(join(setsPath, set))
        modelsPath = join(str(self.caseFile["filesPath"]), "DNNModels_" + str(self.retrainMode))
        if not exists(modelsPath):
            os.makedirs(modelsPath)
        modelsList = os.listdir(modelsPath)
        for set in modelsList:
            if set.startswith(self.retrainMode):
                os.remove(join(modelsPath, set))
            if set.startswith("Report_" + self.retrainMode):
                os.remove(join(modelsPath, set))

    def explain(self):
        self.RCC = "TR"
        self.updateCaseFile()
        self.selectLayer()
        xplainModule.run(self.caseFile)

    def getParams(self):
        self.selectLayer()
        # self.selectedLayer = "Layer9"

        # clsData = torch.load(join(self.outputPathOriginal, self.faceSubset, self.RCC, "ClusterAnalysis_" +
        # self.clustMode, self.selectedLayer + ".pt"), map_location=torch.device('cpu'))
        clsData = torch.load(
            join(self.outputPathOriginal, self.RCC, "ClusterAnalysis_" + self.clustMode, self.selectedLayer + ".pt"),
            map_location=torch.device('cpu'))
        # clsParam = np.load(join(self.outputPathOriginal, self.faceSubset, "clustersParamData.npy"), allow_pickle=True)
        # print(clsParam)
        # return
        #paramsModule.getParams(self.testCSV, self.testDataNpy, join(self.caseFile["outputPathOriginal"], self.faceSubset, self.RCC, self.selectedLayer+"_WICD"), clsData, join(self.caseFile["filesPath"], "DT_MC_RCC.csv"))
        # paramsModule.getParams(self.improveCSV,join(self.outputPath, "IEEPackage", "ieeimprove.npy"), join(self.outputPathOriginal, self.RCC, self.selectedLayer+"_WICD"), clsData, join(self.caseFile["filesPath"], "DT.csv"))
        paramsModule.getParams(self.testCSV, self.testDataNpy,
                               join(self.outputPathOriginal, self.RCC, self.selectedLayer + "_WICD"), clsData,
                               join(self.caseFile["filesPath"], "DT_MC_CC.csv"))
        # paramsModule.getParams(self.trainCSV,self.trainDataNpy, join(self.outputPathOriginal, self.RCC, self.selectedLayer+"_WICD"), clsData, join(self.caseFile["filesPath"], "DT.csv"))

    def updateCaseFile(self):
        if self.datasetName != "SAP":
            if self.iee_version:
                self.caseFile["iee_version"] = self.iee_version
            else:
                self.caseFile["iee_version"] = 0
            self.caseFile["KP"] = self.KP
            self.caseFile["FLD"] = self.FLD
            self.caseFile["RCC"] = self.RCC
            if "DNN" not in self.caseFile:
                self.caseFile["DNN"] = self.DNN
            #if "DNN2" not in self.caseFile:
            #    self.modelPath = join(self.outputPath, "DNNModels", self.modelName_S)
                #print("DNN2", self.modelPath)
            #    self.loadDNN(dnnModels.AlexNetIEE(self.numClass))
            #    self.caseFile["DNN2"] = self.DNN
            #self.modelPath = join(self.outputPath, "DNNModels", self.modelName_R)
            #print("DNN1", self.modelPath)
            if self.datasetName == "HPD":
                self.loadDNN(dnnModels.AlexNetIEE(self.numClass))
            if self.datasetName == "FLD":
                self.loadDNN(dnnModels.KPNet())
            self.caseFile["Alex"] = self.Alex
            self.caseFile["Epochs"] = self.Epochs
            self.caseFile["imgExt"] = self.imgExt
            self.caseFile["metric"] = self.metric
            self.caseFile["layers"] = self.layers
            self.caseFile["testCSV"] = self.testCSV
            self.caseFile["trainCSV"] = self.trainCSV
            self.caseFile["improveCSV"] = self.improveCSV
            self.caseFile["expNum1"] = self.expNumber
            self.caseFile["numClass"] = self.numClass
            self.caseFile["expNum2"] = self.expNumber2
            self.caseFile["modelPath"] = self.modelPath
            self.caseFile["maxCluster"] = self.maxClust
            self.caseFile["batchSize"] = self.batchSize
            self.caseFile["outputPath"] = self.outputPath
            self.caseFile["faceSubset"] = self.faceSubset
            self.caseFile["clustMode"] = self.clustMode
            self.caseFile["assignMode"] = self.assignMode
            self.caseFile["datasetName"] = self.datasetName
            self.caseFile["retrainMode"] = self.retrainMode
            self.caseFile["retrainMode"] = self.retrainMode
            self.caseFile["testFlag"] = self.saveHMTestFlag
            self.caseFile["testDataSet"] = self.testDataSet
            self.caseFile["testDataNpy"] = self.testDataNpy
            self.caseFile["DataSetsPath"] = self.DataSetsPath
            self.caseFile["scratchFlag"] = self.scratchFlag
            self.caseFile["workersCount"] = self.workersCount
            self.caseFile["trainDataNpy"] = self.trainDataNpy
            self.caseFile["trainDataSet"] = self.trainDataSet
            self.caseFile["trainFlag"] = self.saveHMTrainFlag
            self.caseFile["testDataPath"] = self.testDataPath
            self.caseFile["drawClustFlag"] = self.drawClustFlag
            self.caseFile["selectedLayer"] = self.selectedLayer
            self.caseFile["selectionMode"] = self.selectionMode
            self.caseFile["trainDataPath"] = self.trainDataPath
            self.caseFile["improveDataNpy"] = self.improveDataNpy
            self.caseFile["realDataNpy"] = self.realDataNpy
            self.caseFile["improveDataSet"] = self.improveDataSet
            self.caseFile["improveDataPath"] = self.improveDataPath
            self.caseFile["retrainApproach"] = self.retrainApproach
            self.caseFile["outputPathOriginal"] = self.outputPathOriginal
            self.caseFile["filesPath"] = join(self.outputPath, self.RCC)
            self.dlibPath = join(self.caseFile["outputPath"], "IEEPackage/clsdata/mmod_human_face_detector.dat")
            self.filesPath = self.caseFile["filesPath"]
            self.caseFile["components"] = ["noseridge", "nose", "mouth", "rightbrow", "righteye", "lefteye", "leftbrow"]
            self.caseFile["caseFile"] = join(str(self.caseFile["filesPath"]),
                                             "caseFile_" + self.retrainMode + ".pt")
            assignPath = join(str(self.caseFile["filesPath"]), "ClusterAnalysis_" + self.clustMode, "Assignments",
                              self.assignMode, self.selectionMode)
            if isfile(self.improveCSV):
                # print(self.improveCSV, "exists")
                self.saveImproveFlag = False
            else:
                # print(self.improveCSV, "doesn't exist")
                self.saveImproveFlag = True
            #self.saveImproveFlag = False
            if not exists(assignPath):
                os.makedirs(assignPath)
            self.caseFile["assignPTFile"] = join(assignPath, "clusterwithAssignedImages.pt")
            self.caseFile["assignXLFile"] = join(assignPath, "clusterwithAssignedImages.xlsx")
            self.caseFile["improveRCCDists"] = join(assignPath, "improveRCCDists")
            if not exists(self.caseFile["improveRCCDists"]):
                os.makedirs(self.caseFile["improveRCCDists"])
            torch.save(self.caseFile, self.caseFile["caseFile"])

    def loadDNN(self, net):
        if self.CN:

            saver = tf.compat.v1.train.Saver()
            sess = tf.compat.v1.Session()
            print(self.modelPath)
            #sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, self.modelPath)
            #self.DNN = dnnModels.ConvModel()
        else:
            if torch.cuda.is_available():
                if not self.scratchFlag:
                    weights = torch.load(self.modelPath)
                    #print("Loaded", self.modelPath)
                    if self.Alex:
                        net.load_state_dict(weights)
                    elif self.KP:
                        net.load_state_dict(weights.state_dict())
                net = net.to('cuda')
                net.cuda()
                net.eval()
                self.DNN = net
            else:
                if not self.scratchFlag:
                    weights = torch.load(self.modelPath, map_location=torch.device('cpu'))
                    #print("Loaded", self.modelPath)
                    if self.Alex:
                        net.load_state_dict(weights)
                    elif self.KP:
                        net.load_state_dict(weights.state_dict())
                net.eval()
                self.DNN = net

    def selectLayer(self):
        self.selectedLayer = None
        minAvgWICD = [0] * len(self.layers)
        i = 0
        clsPath = join(self.caseFile["filesPath"], "ClusterAnalysis_" + str(self.clustMode))
        for layerX in self.layers:
            clsFile = join(clsPath, layerX + ".pt")
            if torch.cuda.is_available():
                clsData = torch.load(clsFile)
            else:
                clsData = torch.load(clsFile, map_location=torch.device('cpu'))
            # minAvgICD[i] = clsData["avgLayer"]
            minAvgWICD[i] = clsData["WeightedavgLayer"]
            minAvgWICD[0] = 1e9
            i += 1
        indxW = min(enumerate(minAvgWICD), key=itemgetter(1))[0]
        self.selectedLayer = self.layers[indxW]
        print("Selected Layer based on ", self.selectionMode, " is ", str(self.selectedLayer))
        # print(minAvgWICD[indxW])
        # print(minAvgWICD)
        selectedClsFile = join(clsPath, self.selectedLayer + ".pt")
        self.caseFile["clsPath"] = str(selectedClsFile)
        self.caseFile["layerIndex"] = int(self.selectedLayer.replace("Layer", ""))
        self.caseFile["selectedLayer"] = self.selectedLayer
        # dirPath = join(str(self.caseFile["filesPath"]), str(self.selectedLayer) + "_" + str(self.selectionMode))
        # if exists(dirPath):
        #    shutil.rmtree(dirPath)
        # shutil.copytree(join(clsPath, self.selectedLayer), dirPath)
        self.updateCaseFile()

    def retrainDNN(self):
        if self.retrainFlag:
            if self.retrieveAccuracy is not None:
                self.caseFile["retrieveAccuracy"] = self.retrieveAccuracy
            if self.retrainSet is not None:
                self.caseFile["retrainSet"] = self.retrainSet
            self.updateCaseFile()
            retrainModule.run(self.caseFile)
            self.caseFile = torch.load(self.caseFile["caseFile"])
            self.updateCaseFile()
        else:
            print("Retraining module is disabled.")

    def assignImages(self):
        if self.assignFlag:
            # if isfile(self.caseFile["assignPTFile"]):
            #    os.remove(self.caseFile["assignPTFile"])
            # if isfile(self.caseFile["assignXLFile"]):
            #    os.remove(self.caseFile["assignXLFile"])
            self.drawAssignFlag = True
            assignments = [self.assignMode]
            for assMode in assignments:
                print("Assignment Strategy being applied:", assMode)
                self.ResultDict[assMode] = {}
                if self.KP:
                    if not exists(join(self.improveDataPath)):
                        os.mkdir(join(self.improveDataPath))
                    improvCounter = len(os.listdir(join(self.improveDataPath)))
                    if improvCounter < 50:
                        exportImprovFlag = True
                    else:
                        exportImprovFlag = False
                    if exportImprovFlag:
                        print("Exporting improvement images")
                        # ieeDV.exportIEEimages(self.improveDataNpy, self.improveDataSet, join(self.improveDataPath), False, "I")
                else:
                    self.improveDataNpy = None
                    self.trainDataNpy = None
                    self.testDataNpy = None
                print("Performing Assignment at the selected layer: " + str(self.selectedLayer))
                self.caseFile["retrainList"], retrainLength = assignModule.getFolderSize(
                    self.caseFile["improveDataPath"])
                caseFile = assignModule.run(self.caseFile)
                self.caseFile = caseFile
                self.updateCaseFile()
        self.updateCaseFile()

    def generateClusters(self):
        if self.clustFlag:
            clustModes = ['ICDWard', 'ICDAvg', 'DunnWard', 'DunnAvg', 'SWard', 'SAvg', 'DunnICDWard', 'DunnICDAvg',
                          'DBIWard', 'DBIAvg', 'WICDWard', 'WICDAvg']
            clustModes = [self.clustMode]
            for clustMode in clustModes:
                self.clustMode = clustMode
                layerList = list()
                self.updateCaseFile()
                for layerX in self.layers:
                    layerClust = join(self.caseFile["filesPath"], "ClusterAnalysis_" + str(self.clustMode),
                                      layerX + ".pt")
                    if not isfile(layerClust):
                        caseFile = clusterModule.run(self.caseFile)
                        self.caseFile = caseFile
                    else:
                        layerList.append(layerX)
                print(len(layerList), "Layers cluster files exist.")
                if self.drawClustFlag:
                    # clusterModule.drawClusters(self.caseFile, join(self.caseFile["outputPath"], "KP"), self.DNN)
                    clusterModule.drawClusters(self.caseFile, self.testDataPath, self.DNN)
        self.selectLayer()

    def generateHMDistances(self):
        if self.computeFlag:
            layerList = list()
            self.updateCaseFile()
            for layer in self.layers:
                layerDist = join(self.caseFile["filesPath"], layer + "HMDistance.xlsx")
                if not isfile(layerDist):
                    print(len(layerList), "Layers heatmaps-distance files exists.")
                    print("Computing", layer)
                    HeatmapModule.computeDistanceSheets(layer, self.caseFile)
                else:
                    layerList.append(layer)
            print(len(layerList), "Layers heatmaps-distance files exists.")

    def generateHeatmaps(self):
        # self.saveHMFlag = False
        if self.saveHMFlag:
            if self.saveHMTrainFlag:
                if self.Alex:
                    self.trainDataNpy = self.trainDataSet
                self.updateCaseFile()
                HeatmapModule.saveHeatmaps(self.caseFile, "Train")

            if self.saveHMTestFlag:
                if self.Alex:
                    self.testDataNpy = self.testDataSet
                self.updateCaseFile()
                HeatmapModule.saveHeatmaps(self.caseFile, "Test")

    def TL(self):
        _, DNN = alexTrain(epochNum, newTrainDataSet, bestModelPath, DNN, None)
        DNN = loadDNN(caseFile, bestModelPath)
        testAccuracy, resultDictNew = alexTest(bestModelPath, testSet, resultDict, datasetName, DNN, False, None)
        print(testAccuracy.item())
        test.append(testAccuracy.item())

    def train(self):
        retrainModule.genericTrain(self.outputPath, self.datasetName, 100)

    def saveResult(self):
        self.updateCaseFile()
        counter = 0
        if self.saveTrainFlag:
            print("Processing TrainingSet files....")
            if self.KP:
                predictor = self.trainPredict
                counter, _ = predictor.predict(self.trainDataSet, None, self.trainDataPath, True, self.trainCSV, 1,
                                               True, None)
            if self.Alex:
                retrainModule.alexTest(self.modelPath, self.trainDataSet, None, self.datasetName, self.DNN, True,
                                       self.trainCSV)
                # testModule.testErrorAlexNet(self.DNN, self.caseFile, self.trainDataSet, self.saveTrainFlag, self.trainCSV)
        if self.saveTestFlag:
            print("Processing TestSet files....")
            if self.KP:
                predictor2 = self.testPredict

                #predictor2.predict(self.testDataSet, None, self.testDataPath, True, self.testCSV, counter, True, None)
                predictor2.predict(self.testDataSet, None, self.testDataPath, True, self.testCSV, 1, True, None)
            if self.Alex:

                retrainModule.alexTest(self.modelPath, self.testDataSet, None, self.datasetName, self.DNN, True,
                                       self.testCSV)
                # testModule.testErrorAlexNet(self.DNN, self.caseFile, self.testDataSet, self.saveTestFlag, self.testCSV)
        if self.saveImproveFlag:
            print("Processing ImprovementSet files....")
            if self.KP:
                predictor2 = self.improvePredict
                if counter == 0:
                    counter = 1
                predictor2.predict(self.improveDataSet, None, self.improveDataPath, True, self.improveCSV, counter,
                                   False, None)
            if self.Alex:
                retrainModule.alexTest(self.modelPath, self.improveDataSet, None, self.datasetName, self.DNN, True,
                                       self.improveCSV)
                #testModule.testErrorAlexNet(self.DNN, self.caseFile, self.improveDataSet, self.saveImproveFlag,
                #                            self.improveCSV)
        testError = 0
        testError2 = 0
        testTotal = 0
        trainError = 0
        trainError2 = 0
        trainTotal = 0
        improveTotal = 0
        improveError = 0
        improveError2 = 0
        imageList = pd.read_csv(self.testCSV)
        for index, row in imageList.iterrows():
            testTotal += 1
            if row["result"] == "Wrong":
                if self.datasetName == "FLD":
                    testError2 += 1
                    #if row["worst_component"] == self.faceSubset:
                    #    testError += 1
                    testError += 1
                else:
                    testError += 1
        imageList = pd.read_csv(self.trainCSV)
        for index, row in imageList.iterrows():
            trainTotal += 1
            if row["result"] == "Wrong":
                if self.datasetName == "FLD":
                    trainError2 += 1
                    if row["worst_component"] == self.faceSubset:
                        trainError += 1
                else:
                    trainError += 1
        imageList = pd.read_csv(self.improveCSV)
        for index, row in imageList.iterrows():
            improveTotal += 1
            if row["result"] == "Wrong":
                if self.datasetName == "IEEKP":
                    improveError2 += 1
                    if row["worst_component"] == self.faceSubset:
                        improveError += 1
                else:
                    improveError += 1
        self.caseFile["trainError"] = trainError
        self.caseFile["trainTotal"] = trainTotal
        self.caseFile["testError"] = testError
        self.caseFile["testTotal"] = testTotal
        self.caseFile["improveTotal"] = improveTotal
        self.caseFile["improveError"] = improveError
        if self.datasetName == "FLD":
            self.caseFile[self.faceSubset] = {}
            self.caseFile[self.faceSubset]["trainError"] = trainError
            self.caseFile[self.faceSubset]["testError"] = testError
            self.caseFile[self.faceSubset]["improveError"] = improveError
            self.caseFile["trainError"] = trainError2
            self.caseFile["testError"] = testError2
            self.caseFile["improveError"] = improveError2

        if self.saveHMTrainFlag:
            self.maxClust = int(trainError + testError / 2)
        else:
            self.maxClust = int(testError / 2)
        if self.maxClust < 2:
            self.maxClust = 2
        if self.maxClust > 500:
            self.maxClust = 500
        print("TrainingSet Size:", trainTotal, "Total misclassified images:", trainError,
              "TrainingSet accuracy:", str(((1 - (trainError / trainTotal)) * 100.00))[0:6] + "%")
        print("TestSet Size:", testTotal, "Total misclassified images:", testError,
              "TestSet accuracy:", str(((1 - (testError / testTotal)) * 100.00))[0:6] + "%")
        print("ImprovementSet Size:", improveTotal, "Total misclassified images:", improveError,
              "ImprovementSet accuracy:", str(((1 - (improveError / improveTotal)) * 100.00))[0:6] + "%")
        # print("ImprovementSet Size:", improveTotal, "ImprovementSet accuracy:",
        #      str((1 - (self.caseFile["improveError"] / self.caseFile["improveTotal"])) * 100.00)[0:6] + "%")

    def injectFaults(self):
        self.saveResult()
        n = 10 #faults per class
        if self.caseFile["datasetName"] == "GD" or self.caseFile["datasetName"] == "OC":
            classes, faults = self.injectBlurNoise(n)
        else:  # HPD / FLD
            #classes, faults = self.injectBlurNoise(n)
            self.injectHands()
        exit()
        classes, faults = self.bagFaults(classes, faults, n)

        self.recomputeTestSet()

        classes, faults = injectFaults.setClassesFaults(self.caseFile)
        classes, faults = injectFaults.getFaults(self.caseFile["testCSV"], classes, faults)
        print("faults per class", classes)
        print("total faults", faults)
        return

    def injectHands(self):
        import pathlib as pl
        imgsPath = join(self.outputPath, "DataSets", "HandsOnFace")
        newDir = join(self.outputPath, "DataSets", "Labelled-HOF")
        labelDir = join(self.outputPath, "DataSets", "NPY-HOF")
        images = list()
        for src_dir, dirs, files in os.walk(imgsPath):
            for file_ in files:
                if (file_.endswith(".jpg")) or (file_.endswith(".png")) or (file_.endswith(".ppm")):
                    images.append(join(src_dir, file_))
        idx = 1
        for img in images:
            if simulatorModule.processImage(img, join(pl.Path(__file__).parent.resolve(), "IEEPackage", "clsdata",
                                               "mmod_human_face_detector.dat")):
                DNNResult, pred, label = simulatorModule.doImage(img, self.caseFile, None)
                if not exists(join(newDir, label)):
                    makedirs(join(newDir, label))
                if not DNNResult:
                    shutil.copy(img, join(newDir, label, "H" + str(idx) + ".png"))
                    shutil.copy(img.split(".png")[0] + ".npy", join(newDir, label, "H" + str(idx) + ".npy"))
            idx += 1
        return

    def injectBlurNoise(self, n):
        classes, faults = injectFaults.setClassesFaults(self.caseFile)
        classes, faults = injectFaults.getFaults(self.caseFile["testCSV"], classes, faults)
        print("injecting faults from TestSet")
        classes, faults = injectFaults.inject(self.caseFile, self.caseFile["testCSV"], faults, classes,
                                              self.caseFile["testDataPath"], n)
        useImprove = False
        for label in classes:
            for fault in faults:
                if classes[label][fault] < n:
                    useImprove = True
        if useImprove:
            print("injecting faults from ImprovementSet")
            classes, faults = injectFaults.inject(self.caseFile, self.caseFile["improveCSV"], faults, classes,
                                                  self.caseFile["testDataPath"], n)
        return classes, faults
    def bagFaults(self, classes, faults, n):
        bagging = False
        for label in classes:
            for fault in faults:
                if 0 < classes[label][fault] < n:
                    bagging = True
        if bagging:
            self.recomputeTestSet()
            classes, faults = injectFaults.bagFaults(self.caseFile["testCSV"], classes, faults,
                                                     self.caseFile["testDataPath"], n)
        return classes, faults
    def recomputeTestSet(self):
        os.remove(self.caseFile["testCSV"])
        dataTransformer = setupTransformer(self.datasetName)
        transformedData = PathImageFolder(root=self.testDataPath, transform=dataTransformer)
        self.testDataSet = torch.utils.data.DataLoader(transformedData, batch_size=self.batchSize, shuffle=True,
                                                       num_workers=self.workersCount)
        self.saveTestFlag = True
        self.saveResult()