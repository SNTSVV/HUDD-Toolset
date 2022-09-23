#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#

from imports import time, os, math, torch, np, pd, Variable, Image, setupTransformer, normalize, join, exists, isfile, \
    basename, isdir, dirname, entropy, pairwise_distances
import dataSupplier as DS
from matplotlib import pyplot


def computeDistanceSheets(layerX, caseFile):
    print("Calculating HM distances ....")
    calcAndSaveHeatmapDistances(layerX, caseFile["filesPath"], "HMDistance.xlsx", caseFile["metric"])
    print("Calculating of HM distances is completed")


def saveHeatmaps(caseFile, prefix):
    dnn = caseFile["DNN"]
    datasetName = caseFile["datasetName"]
    retrainData = caseFile["testDataNpy"]
    csvPath = caseFile["testCSV"]
    if prefix == "Train":
        retrainData = caseFile["trainDataNpy"]
        csvPath = caseFile["trainCSV"]
    elif prefix == "Test":
        retrainData = caseFile["testDataNpy"]
        csvPath = caseFile["testCSV"]
    elif prefix == "I":
        retrainData = caseFile["improveDataNpy"]
        csvPath = caseFile["improveCSV"]
    start = time.time()
    imgListX = []
    if datasetName != 'FLD':
        imgClasses = retrainData.dataset.classes
        imageList = pd.read_csv(csvPath, names=["image", "result", "expected", "predicted"].append(imgClasses))
        for index, row in imageList.iterrows():
            if row["result"] == "Wrong":
                imgListX.append(row["image"])
        print("The test result file contains " + str(len(imageList)) + " records")
        print("Saving heatmaps for " + str(len(imgListX)) + " images")
        csvPath = imgListX
    calcTestImagesHeatmap(caseFile, retrainData, csvPath, dnn, prefix + "_")
    end = time.time()
    print("Total time consumption of operation Saving Heatmaps is " + str((end - start) / 60.0) + " minutes.")


def generateActivations(inputImage, model, dataSetName: str, outputPath: str, visulize: bool):
    with torch.no_grad():
        image = Image.open(inputImage)
        transformer = setupTransformer(dataSetName)
        imageTensor = transformer(image).float()
        imageTensor = imageTensor.unsqueeze_(0)
        imageTensor.detach()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            imageTensor, model = imageTensor.cuda(), model.cuda()
        ### enable instrumenation of model
        for i in range(0, len(model.features)):
            model.features[i].register_forward_hook(forward_hook)

        for i in range(0, len(model.classifier)):
            model.classifier[i].register_forward_hook(forward_hook)

        inputToModel = Variable(imageTensor)
        model.forward(inputToModel)
        k = 0
        sizee = len(model.features) + len(model.classifier)
        layersHM = [0] * sizee
        for i in range(0, len(model.features)):
            layersHM[k] = model.features[i].Y.detach()
            layersHM[k].detach()
            k = k + 1
        for i in range(0, len(model.classifier)):
            layersHM[k] = model.classifier[i].Y.detach()
            layersHM[k].detach()
            k = k + 1
        if visulize:
            visualizeHeatMap(inputImage, layersHM, outputPath)
        del model
        del imageTensor
        return layersHM


def safeHM(HMFile, layer, inputImage, model, dataSetName, outputPath, visulize, area, npyPath, imgExt, FLD):
    if exists(HMFile):
        if not torch.cuda.is_available():
            HM = torch.load(HMFile, map_location='cpu')
        else:
            HMtemp = torch.load(HMFile, map_location='cuda:0')
            HM = HMtemp.cuda()
    else:
        #print("not found")
        HMtot, _ = generateHeatMap(inputImage, model, dataSetName, outputPath, visulize, area, npyPath, imgExt, FLD)
        HM = HMtot[layer]
        # HM = HM.cpu()
        # print(HM)
        if not isdir(dirname(HMFile)):
            os.makedirs(dirname(HMFile))
        torch.save(HM, HMFile)
    return HM


def IEEmapper(area, npyPath, filePath, FLD):
    #print(npyPath, filePath)
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
    # makeFolder(outputPath)
    dataset = np.load(npyPath, allow_pickle=True)
    dataset = dataset.item()
    x_data = dataset["data"]
    y_data = dataset["label"]
    x_data = x_data.astype(np.float32)
    x_data = x_data / 255.
    x_data = x_data[:, np.newaxis]
    # print(inputImage)
    imgSource = basename(dirname(filePath))
    trainImage = str(basename(filePath).split(".")[0])
    # print(trainImage)
    if imgSource == "TrainingSet":
        fileName = int(trainImage)
    if imgSource == "ImprovementSet":
        fileName = int(trainImage.split("I")[1])
    elif imgSource == "TestSet":
        fileName = int(trainImage)
        #if int(FLD) == 1:
        #    fileName = str(fileName - 23041)
        #if int(FLD) == 2:
        #    fileName = str(fileName - 16013)
        #if int(FLD) == 3:
            #fileName = str(fileName + 1)
        fileName = str(fileName + 1)
    else:
        fileName = int()
    # print(trainImage)
    #print(trainImage)
    inputs = x_data[int(fileName) - 1]
    return KParray, inputs, dataset, y_data[int(fileName) - 1]


def generateHeatMap(inputImage, model, dataSetName, outputPath, visulize, area, npyPath, imgExt, FLD):
    model = model.eval()
    if dataSetName == "FLD":
        # imageName = "I" + str(counter) + ".pt"
        # savePath = join(outputPath, imageName)
        # if not isfile(savePath):

        KParray, inputs, dataset, labels_gt = IEEmapper(area, npyPath, inputImage, FLD)
        transformer = setupTransformer(dataSetName)
        inputs = transformer(inputs)
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            model = model.cuda()
        else:
            inputs = Variable(inputs)
        model = ieeRegister(model)
        predict = model(inputs.unsqueeze(0).float())
        predict_cpu = predict.cpu()
        predict_cpu = predict_cpu.detach().numpy()
        predict_xy1 = DS.transfer_target(predict_cpu, n_points=DS.n_points)
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
        predict_cpu = ieeBackKP(predict_cpu, worst_label)
        # predict_cpu = HeatmapModule.ieeBackParts(predict_cpu, area)
        tAF = torch.from_numpy(predict_cpu[0]).type(torch.FloatTensor)
        if torch.cuda.is_available():
            tAF = Variable(tAF).cuda()
        else:
            tAF = Variable(tAF).cpu()
        model.relprop(tAF)
        layersHM = returnHeatmap(model, False, True)
        entropy_ = None
        # layerIndex = int(layer.replace("Layer", ""))
        # torch.save(heatmaps[layerIndex], savePath)
    else:
        with torch.no_grad():
            image = Image.open(inputImage)
            transformer = setupTransformer(dataSetName)
            imageTensor = transformer(image).float()
            imageTensor = imageTensor.unsqueeze_(0)
            imageTensor.detach()
            if torch.cuda.is_available():
                # torch.cuda.empty_cache()
                imageTensor = imageTensor.cuda()
                # model = model.cuda()
            else:
                imageTensor = imageTensor.cpu()
                # model = model.cpu()
            ### enable instrumenation of model
            for i in range(0, len(model.features)):
                model.features[i].register_forward_hook(forward_hook)

            for i in range(0, len(model.classifier)):
                model.classifier[i].register_forward_hook(forward_hook)

            inputToModel = Variable(imageTensor)
            outputFromModel = model.forward(inputToModel)
            outputFromModel = outputFromModel.detach().cpu().numpy()
            # if torch.cuda.is_available():
            #    inputToModel = inputToModel.cuda()
            best = outputFromModel.argmax()
            # outputProb = outputFromModel[0][best]
            #print(outputFromModel)
            en = np.exp(outputFromModel) / np.sum(np.exp(outputFromModel))
            en = en[0]
            en /= sum(en)
            entropy_ = 0
            for cl in en:
                if cl != 0:
                    entropy_ = entropy_+(-1*cl * math.log2(cl))

            en = np.exp(outputFromModel) / np.sum(np.exp(outputFromModel))
            en = en[0]
            entropy_ = entropy(en, base=10) ##fixme base=num_classes
            #print(outputFromModel)
            #normalizedOutput = (outputFromModel - outputFromModel.min()) / (outputFromModel.max() - outputFromModel.min())
            #print(normalizedOutput)
            #logOutput = np.log(normalizedOutput)
            #print(logOutput)
            #entropy = -1 * np.sum(normalizedOutput*logOutput)
            #print(entropy)
            ## the following code only calculate the postive relevance. why an image is classified to a certain class.
            # this is why the relevance of output features except the highest one is set to zero.
            for i in range(0, best):
                outputFromModel[0][i] = 0
            for i in range(best + 1, len(outputFromModel[0])):
                outputFromModel[0][i] = 0
            tAF = torch.from_numpy(outputFromModel[0]).type(torch.FloatTensor)
            # tAF = Variable(tAF).cpu()
            if torch.cuda.is_available():
                tAF = Variable(tAF).cuda()
            else:
                tAF = Variable(tAF).cpu()
            model.relprop(tAF)
            layersHM = returnHeatmap(model, True, True)
            if visulize:
                #print("here", outputPath)
                visualizeHeatMap(inputImage, layersHM, outputPath)
            del model
            del imageTensor
            del tAF
    return layersHM, entropy_


def generateHeatMapOfLayer(inputImage, model, dataSetName: str, layerIndex: int):
    with torch.no_grad():
        image = Image.open(inputImage)
        transformer = setupTransformer(dataSetName)
        imageTensor = transformer(image).float()
        imageTensor = imageTensor.unsqueeze_(0)
        imageTensor.detach()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            imageTensor, model = imageTensor.cuda(), model.cuda()
        ### enable instrumenation of model
        for i in range(0, len(model.features)):
            model.features[i].register_forward_hook(forward_hook)

        for i in range(0, len(model.classifier)):
            model.classifier[i].register_forward_hook(forward_hook)

        inputToModel = Variable(imageTensor)
        outputFromModel = model.forward(inputToModel)
        outputFromModel = outputFromModel.detach().cpu().numpy()
        if torch.cuda.is_available():
            inputToModel = inputToModel.cuda()
        best = outputFromModel.argmax()
        outputProb = outputFromModel[0][best]

        ## the following code only calculate the postive relevance. why an image is classified to a certain class.
        # this is why the relevance of output features except the highest one is set to zero.
        for i in range(0, best):
            outputFromModel[0][i] = 0

        for i in range(best + 1, len(outputFromModel[0])):
            outputFromModel[0][i] = 0

        tAF = torch.from_numpy(outputFromModel[0]).type(torch.FloatTensor)
        # tAF = Variable(tAF).cpu()
        if torch.cuda.is_available():
            tAF = Variable(tAF).cuda()
        else:
            tAF = Variable(tAF).cpu()
        model.relprop(tAF)

        if layerIndex < len(model.features):
            heatmap = model.features[layerIndex].HM.detach()
        else:
            heatmap = model.classifier[layerIndex - len(model.features)].HM.detach()
        del model
        del imageTensor
        del tAF
        return heatmap


def forward_hook(self, input, output):
    # print("forward hook..")
    self.X = input[0]
    self.Y = output


def ieeRegister(model):
    model.conv2d_1.register_forward_hook(forward_hook)
    model.conv2d_2.register_forward_hook(forward_hook)
    model.maxpool_1.register_forward_hook(forward_hook)
    model.conv2d_3.register_forward_hook(forward_hook)
    model.conv2d_4.register_forward_hook(forward_hook)
    model.maxpool_2.register_forward_hook(forward_hook)
    model.conv2d_5.register_forward_hook(forward_hook)
    model.conv2d_6.register_forward_hook(forward_hook)
    model.conv2d_trans_1.register_forward_hook(forward_hook)
    model.conv2d_trans_2.register_forward_hook(forward_hook)
    return model


def ieeBackKP(predict_cpu, KPindex):
    for i in range(0, len(predict_cpu)):
        for j in range(0, len(predict_cpu[i])):
            if j != KPindex:
                predict_cpu[i][j] = 0
    return predict_cpu


def ieeBackParts(predict_cpu, area):
    rightbrow = [2, 3]
    leftbrow = [0, 1]
    mouth = [23, 24, 25, 26]
    righteye = [16, 17, 18, 19, 20, 21, 22]
    lefteye = [9, 10, 11, 12, 13, 14, 15]
    noseridge = [4, 5]
    nose = [6, 7, 8]
    for i in range(0, len(predict_cpu)):
        for j in range(0, len(predict_cpu[i])):
            if (area == "mouth"):
                if (mouth.count(j) < 1):
                    predict_cpu[i][j] = 0
            elif (area == "nose"):
                if (nose.count(j) < 1):
                    predict_cpu[i][j] = 0
            elif (area == "righteye"):
                if (righteye.count(j) < 1):
                    predict_cpu[i][j] = 0
            elif (area == "lefteye"):
                if (lefteye.count(j) < 1):
                    predict_cpu[i][j] = 0
            elif (area == "rightbrow"):
                if (rightbrow.count(j) < 1):
                    predict_cpu[i][j] = 0
            elif (area == "leftbrow"):
                if (leftbrow.count(j) < 1):
                    predict_cpu[i][j] = 0
            elif (area == "noseridge"):
                if (noseridge.count(j) < 1):
                    predict_cpu[i][j] = 0
    return predict_cpu


def returnActivations(model):
    heatmaps = [0] * 10
    heatmaps[0] = model.conv2d_1.Y.detach()
    print(heatmaps[0])
    heatmaps[1] = model.conv2d_2.Y.detach()
    heatmaps[2] = model.maxpool_1.Y.detach()
    heatmaps[3] = model.conv2d_3.Y.detach()
    heatmaps[4] = model.conv2d_4.Y.detach()
    heatmaps[5] = model.maxpool_2.Y.detach()
    heatmaps[6] = model.conv2d_5.Y.detach()
    heatmaps[7] = model.conv2d_6.Y.detach()
    heatmaps[8] = model.conv2d_trans_1.Y.detach()
    heatmaps[9] = model.conv2d_trans_2.Y.detach()

    # heatmaps.size() = [3*128*96]
    print(heatmaps[0].size)
    heatmapVis = heatmaps[0][0].data.cpu().numpy()
    # heatmapVis = heatmapVis.astype('uint8')
    print(heatmapVis.shape)
    # heatmapVis = np.absolute(heatmapVis)
    # heatmapVis = 1000000 * heatmapVis
    # img1 = Image.open("/users/hazem.fahmy/Documents/HPC/IEE/DataSets/TrainingSet/23039.png")
    img2 = Image.fromarray(heatmapVis, "1")
    # img3 = Image.new('L', (img1.width + img2.width, img2.height))
    # img3.paste(img1, (0, 0))
    # img3.paste(img2, (img1.width, 0))
    img2.save("/users/hazem.fahmy/Documents/HPC/IEE/lefteye/_negetive_heatmap.JPEG")
    # heatmapVis = 250 * (heatmapVis-heatmapVis.min()) / (heatmapVis.max()-heatmapVis.min())

    return heatmaps


def returnHeatmap(model, Alex, HM):
    if not Alex:
        heatmaps = [0] * 10
        if HM:
            heatmaps[0] = model.conv2d_1.HM.detach()
            heatmaps[1] = model.conv2d_2.HM.detach()
            heatmaps[2] = model.maxpool_1.HM.detach()
            heatmaps[3] = model.conv2d_3.HM.detach()
            heatmaps[4] = model.conv2d_4.HM.detach()
            heatmaps[5] = model.maxpool_2.HM.detach()
            heatmaps[6] = model.conv2d_5.HM.detach()
            heatmaps[7] = model.conv2d_6.HM.detach()
            heatmaps[8] = model.conv2d_trans_1.HM.detach()
            heatmaps[9] = model.conv2d_trans_2.HM.detach()
        else:
            heatmaps[0] = model.conv2d_1.Y.detach()
            heatmaps[1] = model.conv2d_2.Y.detach()
            heatmaps[2] = model.maxpool_1.Y.detach()
            heatmaps[3] = model.conv2d_3.Y.detach()
            heatmaps[4] = model.conv2d_4.Y.detach()
            heatmaps[5] = model.maxpool_2.Y.detach()
            heatmaps[6] = model.conv2d_5.Y.detach()
            heatmaps[7] = model.conv2d_6.Y.detach()
            heatmaps[8] = model.conv2d_trans_1.Y.detach()
            heatmaps[9] = model.conv2d_trans_2.Y.detach()
    else:
        k = 0
        sizee = len(model.features) + len(model.classifier)
        heatmaps = [0] * sizee
        for i in range(0, len(model.features)):
            if HM:
                heatmaps[k] = model.features[i].HM.detach()
            else:
                heatmaps[k] = model.features[i].Y.detach()
            k += 1
        for i in range(0, len(model.classifier)):
            if HM:
                heatmaps[k] = model.classifier[i].HM.detach()
            else:
                heatmaps[k] = model.classifier[i].Y.detach()
            k += 1
    return heatmaps


def calcTestImagesHeatmap(caseFile, npyPath, csvPath, model, imgSource):
    outputPath = join(caseFile["filesPath"], "Heatmaps")
    DataSetsPath = join(caseFile["outputPath"], "DataSets")
    datasetName = caseFile["datasetName"]
    area = caseFile["faceSubset"]
    layers = caseFile["layers"]
    FLD = caseFile["FLD"]
    imgExt = caseFile["imgExt"]
    if not exists(outputPath):
        os.mkdir(outputPath)
    if (datasetName == "FLD"):
        imageList = pd.read_csv(csvPath)
        dataset = np.load(npyPath, allow_pickle=True)
        dataset = dataset.item()
        x_data = dataset["data"]
        x_data = x_data.astype(np.float32)
        x_data = x_data / 255.
        x_data = x_data[:, np.newaxis]
        cnt1 = 0
        cnt2 = 0
        for index, row in imageList.iterrows():
            cnt2 = 0
            if row["result"] == "Wrong":
                    cnt1 += 1
                    fileName = basename(str(row["image"]))
                    imageName = imgSource + fileName.split(".")[0] + ".pt"
                    saveFlag = False
                    for layerX in layers:
                        DirX = join(outputPath, layerX)
                        if not exists(DirX):
                            os.mkdir(DirX)
                        savePath = join(DirX, imageName)
                        if not isfile(savePath):
                            saveFlag = True
                    if saveFlag:
                        if imgSource == "Train_":
                            filePath = join(DataSetsPath, "TrainingSet", fileName)
                        else:
                            filePath = join(DataSetsPath, "TestSet", fileName)
                        HMtot, _ = generateHeatMap(filePath, model, datasetName, "", False, None, npyPath,
                                                imgExt, FLD)
                        for layerX in layers:
                            savePath = join(outputPath, layerX, imageName)
                            if not isfile(savePath):
                                layerIndex = int(layerX.replace("Layer", ""))
                                torch.save(HMtot[layerIndex], savePath)

            if cnt2 % 1000 == 0:
                print("Generated {} Heatmaps".format(cnt1), end="\r")
    else:
        ##this part need to be modified to save heatmaps by Layer
        cnt1 = 0
        imageList = csvPath
        print("Total heatmaps to collect", len(imageList))
        #print(imageList[0], model, datasetName, outputPath)

        #print()
        for imagePath in imageList:
            imageFileName = basename(imagePath)
            imageClass = imagePath.split(os.sep)[len(imagePath.split(os.sep)) - 2]
            cnt1 = cnt1 + 1
            # print("Checked {} images".format(cnt1), end="\r")
            if imgSource == "Train_":
                inputImage = join(DataSetsPath, "TrainingSet", imageClass, imageFileName)
            elif imgSource == "Test_":
                inputImage = join(DataSetsPath, "TestSet", imageClass, imageFileName)
            else:
                inputImage = join(DataSetsPath, "ImprovementSet", "ImprovementSet", imageClass, imageFileName)
            imageName = imgSource + imageFileName.split(".")[0] + "_" + str(imageClass) + ".pt"
            saveFlag = False
            for layerX in layers:
                DirX = join(outputPath, layerX)
                if not exists(DirX):
                    os.mkdir(DirX)
                savePath = join(DirX, imageName)
                if not isfile(savePath):
                    saveFlag = True
            if saveFlag:
                HMtot, _ = generateHeatMap(inputImage, model, datasetName, join(dirname(DataSetsPath), "T", "Visualize", layerX), False, area, npyPath, imgExt, FLD)
                for layerX in layers:
                    layerIndex = int(layerX.replace("Layer", ""))
                    savePath = join(outputPath, layerX, imageName)
                    torch.save(HMtot[layerIndex], savePath)
                # heatmaps = generateHeatMap(inputImage, model, datasetName, outputPath, False, imgExt)
                # heatmaps = generateActivations(inputImage, model, datasetName, outputPath, True)


            if cnt1 % 1000 == 0:
                print("Generated {} Heatmaps".format(cnt1), end="\r")


def calcRetrainImagesHeatmap(imagesList, reTrainset, datasetName, model, layer, area, inputDir):
    heatmaps = {}
    x = 0
    if (datasetName == "FLD"):
        for idx, (inputsX, labels) in enumerate(reTrainset):
            for inputs in inputsX:
                fileName = imagesList[x]
                inputs = Variable(inputs)
                model = ieeRegister(model)
                predict = model(inputs.unsqueeze(0))
                predict_cpu = predict.cpu()
                predict_cpu = predict_cpu.detach().numpy()
                predict_cpu = ieeBackParts(predict_cpu, area)
                tAF = torch.from_numpy(predict_cpu[0]).type(torch.FloatTensor)
                tAF = Variable(tAF).cpu()
                model.relprop(tAF)
                heatmaps[fileName] = returnHeatmap(layer, model)
                x = x + 1
    else:
        with torch.no_grad():
            cnt1 = 0
            for image in reTrainset.dataset.imgs:
                imagePath = image[0]
                imageFileName = basename(imagePath).split(".")[0]
                imageClass = imagePath.split(os.sep)[len(imagePath.split(os.sep)) - 2]
                img = imageClass + "_" + imageFileName
                cnt1 = cnt1 + 1
                print("Generating hetamp of " + imagePath)
                if cnt1 % 10 == 0:
                    print("Image checked: " + str(cnt1) + "/" + str(len(reTrainset.dataset.imgs)))
                layerIndex = int(layer.replace("Layer", ""))
                heatmaps[img], _ = generateHeatMapOfLayer(imagePath, model, datasetName,
                                                       inputDir + "/" + imageClass + "/" + imageFileName,
                                                       layerIndex)
    return heatmaps


def visualizeHeatMap(image, heatmap, outputImagePath, negetiveHeatMap=False):
    heatmap = heatmap[0:12]
    for i in range(0, len(heatmap)):
        #print(heatmap[i].shape)
        heatmapVis = heatmap[i][0].data.cpu().numpy()
        # heatmapVis = np.absolute(heatmapVis)
        # heatmapVis = 1e9 * (np.absolute(heatmapVis))
        # heatmapVis = 1e9 * heatmapVis
        heatmapVis = 800 * (heatmapVis - heatmapVis.min()) / (heatmapVis.max() - heatmapVis.min())
        heatmapVis = heatmapVis.astype('uint8')
        #print(outputImagePath)
        if not exists(outputImagePath):
            os.makedirs(outputImagePath)

        img1 = Image.open(image)
        img1.save(join(outputImagePath, "Visualize", basename(image)))
        # img2 = Image.fromarray(heatmapVis, 'RGB')
        square = int(math.sqrt(len(heatmapVis)))
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(heatmapVis[ix - 1, :, :], cmap='gray')
                ix += 1
        # show the figure
        pyplot.savefig(
            join(outputImagePath, "Visualize", basename(image) + "_heatmapFeatures" + str(i) + ".JPEG"))
        # for j in range(0, len(heatmapVis)):

        #    img2 = cv2.resize(heatmapVis[j], (256, 256), interpolation=cv2.INTER_CUBIC)
        #    img4 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        # img3 = Image.new('RGB', (img1.width + img2.width, img2.height))
        # img3.paste(img1, (0, 0))
        # img3.paste(img2, (img1.width, 0))
        #    if not negetiveHeatMap:
        #        cv2.imwrite(join(outputImagePath, "Visualize", basename(image) + "_heatmap" + str(i) + "_" + str(j) + ".JPEG"), img4)
        # img2.save(join(outputImagePath, "Visualize", basename(image) + "_heatmap" + str(i) + ".JPEG"))
        #        img1.save(join(outputImagePath, "Visualize", basename(image)))
        # else:

        # img3.save(outputImagePath + "_negetive_heatmap.JPEG")


def calcDisatenceRetrainSetWithTestSet(reTrainSetHeatmaps, testSetHeatMaps, metric):
    with torch.no_grad():
        disatenceRetrainSetWithTestSet = {}
        cnt = 0
        for retrainFile in reTrainSetHeatmaps:
            cnt += cnt + 1
            if cnt % 100 == 0:
                print("distance is calcuated for " + str(cnt) + "/" + str(len(reTrainSetHeatmaps[retrainFile])))
            # exit
            if not (retrainFile in disatenceRetrainSetWithTestSet):
                disatenceRetrainSetWithTestSet[retrainFile] = {}
            for testFile in testSetHeatMaps:
                disatenceRetrainSetWithTestSet[retrainFile][testFile] = doDistance(reTrainSetHeatmaps[retrainFile],
                                                                                   testSetHeatMaps[testFile], metric)
        return disatenceRetrainSetWithTestSet


def collectHeatmaps(outPutPath, layerX):
    allHM = {}
    imgList = []
    index2 = 0
    HMDir = join(outPutPath, "Heatmaps", layerX)
    for file in os.listdir(HMDir):
        if file.endswith(".pt"):
            imgList.append(file)
    for file in imgList:
        if torch.cuda.is_available():
            heatMap = torch.load(join(HMDir, file))
            heatMap.cuda()
        else:
            heatMap = torch.load(join(HMDir, file), map_location='cpu')
        allHM[file.split(".")[0]] = heatMap
        index2 = index2 + 1
        if index2 % 1000 == 0:
            print("Heatmap is collected for " + str(index2) + " images")
    print("Collected " + str(layerX) + " heatmaps")
    return allHM, imgList


def collectHeatmaps_Dir(HMDir):
    allHM = {}
    imgList = []
    index2 = 0
    for file in os.listdir(HMDir):
        if file.endswith(".pt"):
            imgList.append(file)
    for file in imgList:
        if torch.cuda.is_available():
            heatMap = torch.load(join(HMDir, file))
            heatMap.cuda()
        else:
            heatMap = torch.load(join(HMDir, file), map_location='cpu')
        allHM[file.split(".")[0]] = heatMap
        index2 = index2 + 1
        if index2 % 1000 == 0:
            print("Heatmap is collected for " + str(index2) + " images")
    return allHM, imgList


def calcAndSaveHeatmapDistances(layerX, outPutPath: str, outputFile: str, metric):
    start = time.time()
    layerDistances = pd.DataFrame()
    testHM, imgList = collectHeatmaps(outPutPath, layerX)
    c1 = 0
    for file in imgList:
        diff = [-1.0] * len(imgList)
        c2 = 0
        HM1 = testHM[file.split(".")[0]]
        for fileX in imgList:
            if c2 == c1:
                diff[c2] = 0.0
            if c2 > c1:
                HM2 = testHM[fileX.split(".")[0]]
                diff[c2] = doDistance(HM1, HM2, metric)
            c2 = c2 + 1
        c1 = c1 + 1
        for x in range(0, len(diff)):
            if diff[x] == -1.0:
                diff[x] = layerDistances[imgList[x].split(".")[0]][c1 - 1]
        layerDistances[file.split(".")[0]] = diff
        print("Collected distances of " + str(c1) + "/" + str(len(imgList)) + " images", end="\r")
    end = time.time()
    print("Time for one Layer " + str(layerX) + " time cost: " + str((end - start) / 60.0) + " minutes")
    writer = pd.ExcelWriter(join(outPutPath, str(layerX) + outputFile), engine='xlsxwriter')
    writer.book.use_zip64()
    print("collected heatmapDistances of " + str(c1) + " images")
    layerDistances.to_excel(writer)
    writer.close()
    # del allHM


def doDistance(tensor1, tensor2, metric):
    if type(tensor1) is np.ndarray:
        if metric == "Euc":
            return math.sqrt(np.sum(np.power(np.subtract(tensor1, tensor2), 2)))
        elif metric == "Man":
            return None #FIXME
    else:
        if torch.cuda.is_available():
            tensor1 = tensor1.cuda()
            tensor2 = tensor2.cuda()
        else:
            tensor1 = tensor1.cpu()
            tensor2 = tensor2.cpu()
    if metric == "Euc":
        return torch.sqrt(torch.sum(torch.pow(torch.sub(tensor1, tensor2), 2))).item()
    elif metric == "Man":
        return torch.sum(torch.abs(torch.sub(tensor1, tensor2))).item()


def unifyHM(outputPath, layerX):
    newHM = {}
    # if not exists(outputPath + "/" + str(layerX) + "HMDistance.xlsx"):
    testHMPath = join(outputPath, "TestHeatmaps", str(layerX))
    trainHMPath = join(outputPath, "TrainHeatmaps", str(layerX))
    if not False:
        if torch.cuda.is_available():
            HM1 = torch.load(testHMPath)
            HM2 = torch.load(trainHMPath)
        else:
            HM1 = torch.load(testHMPath, map_location={'cpu'})
            HM2 = torch.load(trainHMPath, map_location={'cpu'})
        for img in HM1:
            newHM["Test_" + img] = HM1[img]
        for img in HM2:
            newHM["Train_" + img] = HM2[img]
    return newHM


def saveLoadHM(inputPath, outputPath, maxFiles):
    HMfile = {}
    counter = 0
    for file in os.listdir(inputPath):
        # HMfile[file.split(".")[0]] = generateHeatMap(,None,False)
        counter = counter + 1
        if counter == maxFiles:
            counter = 0
            torch.save(HMfile, join(outputPath, file.split(".")[0] + ".pt"))
            HMfile = {}


def calculate_pixel_distance(coord1, coord2):
    diff = np.square(coord1 - coord2)
    sum_diff = np.sqrt(diff[:, :, 0] + diff[:, :, 1])
    avg = sum_diff.mean()
    return avg, sum_diff


def makeFolder(inputPath):
    if not exists(inputPath):
        os.mkdir(inputPath)
