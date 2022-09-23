#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#
import Helper, RQ2, RQ1
from imports import basename, argparse, os, shutil, join, np, exists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    parser.add_argument('-a', '--action', help='supported actions: test, heatmap, cluster, assign, retrain',
                        required=False)
    parser.add_argument('-m', '--modelName', help='pretrainedWeights.pth Path', required=False)
    parser.add_argument('-o', '--outputPathX', help='Output path for saving the result', required=True)
    parser.add_argument('-sF', '--scratchFlag', help='Number of Classes', required=False)
    parser.add_argument('-n', '--ClusterModeX', help='ICD - WICD - S', required=False)
    parser.add_argument('-cF', '--clustF', help='clustering Flag', required=False)
    parser.add_argument('-dcF', '--drawCF', help='Exporting images Flag', required=False)
    parser.add_argument('-aF', '--assignF', help='Exporting images Flag', required=False)
    parser.add_argument('-daF', '--drawAssignF', help='Exporting images Flag', required=False)
    parser.add_argument('-err', '--errorMarginPixels', help='error Margin Pixels', required=False)
    parser.add_argument('-sub', '--faceSubSet', help='Subset of the face', required=False)
    parser.add_argument('-tl', '--transfer', help='scratch/pretrained', required=False)
    parser.add_argument('-rF', '--retrainF', help='HUDD, BL1, BL2', required=False)
    parser.add_argument('-mode', '--retrainMode', help='HUDD, BL1, BL2', required=False)
    parser.add_argument('-app', '--approach', help='A, B', required=False)
    parser.add_argument('-exp1', '--expNumber', help='Number of retrainings', required=False)
    parser.add_argument('-exp2', '--expNumber2', help='Number of retrainings', required=False)
    parser.add_argument('-ep', '--epoch', help='Number of epochs', required=False)
    parser.add_argument('-ass', '--assignMode', help='ICD - Centroid - Closest - SSE', required=False)
    parser.add_argument('-bs', '--BagSize', help='ICD - Centroid - Closest - SSE', required=False)
    parser.add_argument('-mc', '--maxClust', help='ICD - Centroid - Closest - SSE', required=False)
    parser.add_argument('-ow', '--ow', help='overwrite flag', required=False)
    parser.add_argument('-sel', '--select', help='layer selection mode', required=False)
    parser.add_argument('-fld', '--FLD', help='FLD selection mode', required=False)
    parser.add_argument('-wc', '--workersCount', help='FLD selection mode', required=False)
    parser.add_argument('-batchS', '--batchSize', help='FLD selection mode', required=False)
    parser.add_argument('-cleanF', '--cleanFlag', help='FLD selection mode', required=False)
    parser.add_argument('-rcc', '--rccSource', help='FLD selection mode', required=False)
    parser.add_argument('-numR', '--numRuns', help='FLD selection mode', required=False)
    parser.add_argument('-rA', '--retrieveAccuracy', help='FLD selection mode', required=False)
    parser.add_argument('-rq', '--RQ1A', help='FLD selection mode', required=False)
    parser.add_argument('-rS', '--retrainSet', help='FLD selection mode', required=False)
    parser.add_argument('-HUDD', '--HUDDmode', help='FLD selection mode', required=False)
    parser.add_argument('-iee', '--ieeVersion', default="1", help='iee_sim1, iee_sim2', required=False)
    parser.add_argument('-cls', '--clsNum', default="1", help='iee_sim1, iee_sim2', required=False)
    args = parser.parse_args()
    components = ["noseridge", "nose", "mouth", "rightbrow", "righteye", "lefteye", "leftbrow"]
    HUDD = Helper.Helper(outputPath=args.outputPathX, modelName=args.modelName, workersCount=args.workersCount,
                  batchSize=args.batchSize, metric="Euc", clustFlag=args.clustF, assignFlag=args.assignF,
                  retrainFlag=args.retrainF, retrainMode=args.retrainMode, retrainApproach=args.approach,
                  expNumber=args.expNumber, expNumber2=args.expNumber2, bagSize=args.BagSize,
                 clustMode=args.ClusterModeX, assMode=args.assignMode,
                  overWrite=args.ow, selectionMode=args.select, FLD=args.FLD, cleanFlag=args.cleanFlag,
                  RCC=args.rccSource, scratchFlag=args.scratchFlag, retrieveAccuracy=args.retrieveAccuracy,
                  RQ1A=False, retrainSet=args.retrainSet, drawClustFlag=args.drawCF, ieeVersion=args.ieeVersion, clustNum=args.clsNum)
    #HUDD.updateCaseFile()
    finalResultDict = {}
    datasetName = basename(args.outputPathX)
    # assModes = ["Centroid", "ClosestICD", "jICD", "SSEICD", "ClosestMem"]
    if args.HUDDmode == "HUDD":
        if datasetName == "FLD":
            if args.faceSubSet is None:
                maxSub = 0.0
                for subset in components:
                    print(subset)
                    # ResultDict, _ = HUDD.KPNet(subset)
                    # HUDD.faceSubset = subset
                    # HUDD.updateCaseFile()
                    # HUDD.saveResult()
            else:
                print(args.faceSubSet)
                ResultDict, _ = HUDD.KPNet(args.faceSubSet)
        else:
            ResultDict, _ = HUDD.AlexNet()
        if args.numRuns is None:
            if datasetName == "FLD":
                ResultDict, _ = HUDD.KPNet(components[0])
                HUDD.faceSubSet = components[0]
                # HUDD.saveResult()
            HUDD.retrainDNN()
        else:
            for x in range(0, int(args.numRuns)):
                HUDD.retrainDNN()
    elif args.HUDDmode == "RQ2":
        RQ2.run(HUDD.modelName, HUDD.outputPath, HUDD.datasetName, HUDD.modelPath, HUDD.numClass, HUDD.modelArch)
        if datasetName == "FLD":
            ResultDict, _ = HUDD.KPNet(args.faceSubSet)
        else:
            ResultDict, _ = HUDD.AlexNet()
    elif args.HUDDmode == "concepts":
        #RQ2.run(HUDD.modelName, HUDD.outputPath, HUDD.datasetName, HUDD.modelPath, HUDD.numClass, HUDD.modelArch)
        if datasetName == "FLD":
            ResultDict, _ = HUDD.KPNet(args.faceSubSet)
        else:
            #HUDD.faceSubset = "CC"
            #HUDD.faceSubset = "CC"
            HUDD.faceSubset = "CC"
            HUDD.updateCaseFile()
            HUDD.generateConcepts()
    elif args.HUDDmode == "xplain":
        HUDD.explain()
    elif args.HUDDmode == "params":
        HUDD.faceSubset = args.faceSubSet
        HUDD.getParams()
    elif args.HUDDmode == "HUDD":
        HUDD.KPNet(None)
    elif args.HUDDmode == "finetune":
        HUDD.TLDNN()
    elif args.HUDDmode == "newdata":
        HUDD.generateDataSet()
    elif args.HUDDmode == "train":
        HUDD.train()
    elif args.HUDDmode == "RQ1":
        RQ1.IEERQ1(HUDD.caseFile)
    elif args.HUDDmode == "retrain":
        HUDD.selectLayer()
        HUDD.retrainDNN()
    elif args.HUDDmode == "testModel":
        HUDD.saveResult()
    elif args.HUDDmode == "generateHeatmaps":
        HUDD.generateHeatmaps()
    elif args.HUDDmode == "generateHMDists":
        HUDD.generateHMDistances()
    elif args.HUDDmode == "generateClusters":
        HUDD.generateClusters()
        HUDD.selectLayer()
    elif args.HUDDmode == "assignImages":
        HUDD.selectLayer()
        HUDD.assignImages()
    elif args.HUDDmode == "injectFaults":
        HUDD.injectFaults()