#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#
workersCount = 4
batchSize = 24
import testModule
import Helper
import ieepredict
from imports import os, argparse, join, basename

def run(modelName, outputPath, datasetName, weightPath, numClass, modelArch):
    #loadModel
    #loadTestSet
    #Evaluate
    dataPath = join(outputPath, "DataSets", "TestSet")
    modelPath = join(outputPath, "DNNModels", modelName)
    KPData, AlexData, imagesList = testModule.loadData(dataPath, datasetName, workersCount, batchSize, outputPath, weightPath)
    DNN = testModule.loadDNN(modelPath, modelArch, numClass, False)
    testAccuracy = testModule.testErrorDNN(DNN, KPData, AlexData, None, imagesList, None, None, None, False)
    print("The model " + modelName + " shows TestSet accuracy is: " + str(testAccuracy[0].item())[0:4] + "%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    parser.add_argument('-m', '--modelName', help='model (BL1_0.pth,..)', required=True)
    parser.add_argument('-o', '--outputPathX', help='DirectoryPath', required=True)
    args = parser.parse_args()
    outputPath = args.outputPathX
    datasetName = basename(outputPath)
    FLDflag = False
    if datasetName == "GD":
        numClass = 8
        modelArch = "AlexNet"
    elif datasetName == "OC":
        numClass = 2
        modelArch = "AlexNet"
    elif datasetName == "ASL":
        numClass = 29
        modelArch = "AlexNet"
    elif datasetName == "TS":
        numClass = 43
        modelArch = "AlexNet"
    elif datasetName == "OD":
        numClass = 2
        modelArch = "AlexNet"
    elif datasetName == "AC":
        numClass = 8
        modelArch = "AlexNet"
    elif datasetName == "HPD":
        numClass = 9
        modelArch = "AlexNetIEE"
    elif datasetName == "FLD":
        numClass = 27
        modelArch = "KPNet"
        predictor = predict.IEEPredictor(join(outputPath, "DataSets", "ieetest.npy"),
                                         join(outputPath, "DNNModels", args.modelName), 0)
        simDataSet, _ = predictor.load_data(join(outputPath, "DataSets", "ieetest.npy"))
        counter, _ = predictor.predict(simDataSet, None, outputPath, False, None, 1, False, None)
        FLDflag = True
    if not FLDflag:
        run(args.modelName, args.outputPathX, datasetName, None, numClass, modelArch)