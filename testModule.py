#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
# Modified by Mojtaba Bagherzadeh, m.bagherzadeh@uottawa.ca, University of Ottawa, 2019.
#
import dataSupplier as DataSupply
import dnnModels
from imports import os, math, datasets, Variable, torch, time, Image, np, json, cv2, transforms, setupTransformer, join, \
    isfile


def run(testData, datasetName, workersCount, batchSize, modelPath, modelArch, numClasses, outputPath,
        trainData, subsetArea, errorMargin, nPoints):
    start = time.time()

    testDataPath = join(outputPath, "testData.npy")
    trainDataPath = join(outputPath, "trainData.npy")
    testResultPath = join(outputPath, "testResult.csv")
    trainResultPath = join(outputPath, "trainResult.csv")
    weightPath = join(outputPath, "clsdata", "mmod_human_face_detector.dat")
    scratchFlag = False
    if datasetName == "FLD":
        datasetNameX = datasetName + "TEST"
    else:
        datasetNameX = datasetName
    if testData:
        print("Loading test data from " + testData)
        ieeData, unityData, imgList = loadData(dataPath=testData,
                                               dataSetName=datasetNameX,
                                               workersCount=workersCount,
                                               batchSize=batchSize,
                                               outputPath=testDataPath,
                                               weightPath=weightPath)
        print("Loading model  from " + modelPath)
        print(modelPath)
        print(modelArch)
        dnn = loadDNN(modelPath, modelArch, numClasses, scratchFlag)
        dnn = dnn.eval()
        print("model Loaded")
        testErrorDNN(dnn, ieeData, unityData, testResultPath,
                     imgList, errorMargin, subsetArea, nPoints, True)
        print("Saved results in " + testResultPath)
        ieeData, unityData, imgList = loadData(dataPath=trainData,
                                               dataSetName=datasetNameX,
                                               workersCount=workersCount,
                                               batchSize=batchSize,
                                               outputPath=trainDataPath,
                                               weightPath=weightPath)
        # testErrorDNN(dnn, ieeData, unityData, trainResultPath,
        #             imgList, errorMargin, subsetArea, nPoints, True)
        print("Saved results in " + trainResultPath)
    else:
        print("Test data is missing.")

    end = time.time()
    print("Total time consumption of operation \"Extracting Erronous\" Inputs is " + str(
        (end - start) / 60.0) + " minutes.")

def loadData(dataPath: str, dataSetName: str, workersCount: int, batchSize: int, outputPath, weightPath):
    dataSet = 0
    train_di = 0
    imagesList = 0
    if dataSetName == "FLD":
        ds = DataSupply.DataSupplier(using_gm=False)
        if not isfile(outputPath):
            DataSupply.createData(dataPath, outputPath, weightPath)
        train_di, valid_di, imagesList = ds.get_test_iter(outputPath)  # for test data
    elif dataSetName == "IEETRAIN":
        ds = DataSupply.DataSupplier(using_gm=False)
        if not isfile(outputPath):
            DataSupply.createData(dataPath, outputPath)
        train_di = ds.get_train_iter(outputPath)  # for test data
    else:
        dataTransformer = setupTransformer(dataSetName)
        transformedData = PathImageFolder(root=dataPath, transform=dataTransformer)
        dataSet = torch.utils.data.DataLoader(transformedData, batch_size=batchSize, shuffle=True,
                                              num_workers=workersCount)
    return train_di, dataSet, imagesList


def loadDNN(modelPath, modelArch, numClasses, scratchFlag):
    if modelArch == "AlexNet":
        net = dnnModels.AlexNet(numClasses)
        if torch.cuda.is_available():
            print("Torch is available")
            if not scratchFlag:
                weights = torch.load(modelPath)
                net.load_state_dict(weights)
                print("Pretrained weights loaded")
            net = net.to('cuda')
            net.cuda()
        else:
            if not scratchFlag:
                weights = torch.load(modelPath, map_location=torch.device('cpu'))
                net.load_state_dict(weights)
            net.eval()
    if modelArch == "AlexNetIEE":
        net = dnnModels.AlexNetIEE(numClasses)
        if torch.cuda.is_available():
            print("Torch is available")
            if not scratchFlag:
                weights = torch.load(modelPath)
                net.load_state_dict(weights)
                print("Pretrained weights loaded")
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


def testErrorKPNet(model, ieeData1, ieeData2, outPutFile, errorMargin, subsetArea, imgSource, batchSize):
    correctPredictedCount = 0
    totalInputs = 0
    loopIndex = 1
    model.eval()
    # if saveFlag:
    #    outFile = open(outPutFile, 'w')
    #    outFile.writelines("data,label,result,expected,predicted \r\n")
    for x in range(0, int(len(ieeData1) / batchSize)):
        mini = x * batchSize
        maxi = (x + 1) * batchSize
        if (maxi > len(ieeData1)):
            maxi = len(ieeData1)
        inputData = ieeData1[mini:maxi]
        labelData = ieeData2[mini:maxi]
        loopIndex = loopIndex + 1
        if torch.cuda.is_available():
            inputs = Variable(inputData.cuda())
        else:
            inputs = Variable(inputData)
        predict = model(inputs)
        predict_cpu = predict.cpu()
        predict_cpu = predict_cpu.detach().numpy()
        errorList = (predict_cpu, labelData, subsetArea, errorMargin)
        for i in range(0, len(errorList)):
            # grayImage = cv2.cvtColor(np.array(inputData[totalInputs]), cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(join(outPutFile, imgSource + "_Input_" + str(totalInputs)
            #                                                   + ".png"), img)
            if (errorList[i]):
                outcome = "Wrong"
                # torch.save(ieeData1[totalInputs], join(outPutFile, imgSource + "_Input_" + str(totalInputs)
                #                                               + ".pt"))
                # torch.save(ieeData2[totalInputs], join(outPutFile, imgSource + "_Label_" + str(totalInputs)
                #                                               + ".pt"))
            else:
                # outcome = "Correct"
                correctPredictedCount = correctPredictedCount + 1
            # if saveFlag:
            # outFile.writelines(ieeData1[totalInputs] + "," + ieeData2[totalInputs] + "," + outcome + "," + str(0) + "," + str(0) + "\r\n")
            totalInputs = totalInputs + 1
        print(str(totalInputs / len(ieeData1) * 100.0) + "%")
    print("Predicted {} out of {} correctly".format(correctPredictedCount, totalInputs))
    print("The average accuracy is: {} %".format(100.0 * correctPredictedCount / (float(totalInputs))))
    return 100.0 * correctPredictedCount / (float(totalInputs))


def testErrorAlexNet(model, unityData, saveFlag, outPutFile):
    #print(caseFile["modelPath"])
    print(outPutFile)
    correctPredictedCount = 0
    totalInputs = 0
    loopIndex = 1
    model = model.eval()
    classesStr = ','.join(str(class_) for class_ in unityData.dataset.classes)
    if saveFlag:
        outFile = open(outPutFile, 'w')
        outFile.writelines("image,result,expected,predicted," + classesStr + "\r\n")
    detailResults = []
    counter = 0
    for idx, (batch, classes, paths) in enumerate(
            unityData):  # return a list of inputs, classes, and path based on the batch parameter of the dataloader
        # print("loop " + str(loopIndex))
        print("tested inputs " + str(totalInputs), end="\r")
        loopIndex = loopIndex + 1
        totalInputs += len(batch)
        if torch.cuda.is_available():
            batch, classes = batch.cuda(), classes.cuda()
        batch, classes = Variable(batch), Variable(classes)
        scores = model(batch)
        scores = scores.detach()
        pred = scores.data.max(1)[1]
        correctPredictedCount += pred.eq(classes.data).cpu().sum()
        for i in range(len(batch)):
            if (classes.data[i].eq(pred[i])):
                outcome = "Correct"
            else:
                outcome = "Wrong"
                counter = counter + 1
                # print("Number of erronous images " + str(counter))
            strExpected = unityData.dataset.classes[classes[i]]
            strPred = unityData.dataset.classes[pred[i].item()]
            scoreStr = ','.join([str(score) for score in scores[i].data.tolist()])
            if saveFlag:
                outFile.writelines(paths[i] + "," + outcome + "," + strExpected + "," + strPred + "," + scoreStr[1:len(
                                                                                                            scoreStr) -
                                                                                                                   2] +
                                   "\r\n")

    print("Predicted {} out of {} correctly".format(correctPredictedCount, totalInputs))
    print("The average accuracy is: {} %".format(100.0 * correctPredictedCount / (float(totalInputs))))
    print("Total erronous " + str(counter))
    if saveFlag:
        outFile.close()
    return 100.0 * correctPredictedCount / (float(totalInputs)), detailResults


def testErrorDNN(model, ieeData, unityData, outPutFile, imagesList, errorMargin, subsetArea, nPoints, saveFlag):
    correctPredictedCount = 0
    totalInputs = 0
    loopIndex = 1
    model.eval()
    if (unityData == 0):
        if saveFlag:
            outFile = open(outPutFile, 'w')
            outFile.writelines("image,result,expected,predicted \r\n")
        for idx, (inputs, labels) in enumerate(ieeData):
            loopIndex = loopIndex + 1
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            predict = model(inputs)
            predict_cpu = predict.cpu()
            predict_cpu = predict_cpu.detach().numpy()
            errorList = ieeError(predict_cpu, labels, subsetArea, errorMargin, nPoints)

            for i in range(0, len(errorList)):
                if (errorList[i]):
                    outcome = "Wrong"
                else:
                    outcome = "Correct"
                    correctPredictedCount = correctPredictedCount + 1
                if saveFlag:
                    outFile.writelines(imagesList[totalInputs] + "," + outcome + "," + str(0) + "," + str(0) + "\r\n")
                totalInputs = totalInputs + 1

        print("Predicted {} out of {} correctly".format(correctPredictedCount, totalInputs))
        print("The average accuracy is: {} %".format(100.0 * correctPredictedCount / (float(totalInputs))))
        return 100.0 * correctPredictedCount / (float(totalInputs))

    else:
        classesStr = ','.join(str(class_) for class_ in unityData.dataset.classes)
        for class_ in unityData.dataset.classes:
            print(class_)
        if saveFlag:
            outFile = open(outPutFile, 'w')
            outFile.writelines("image,result,expected,predicted," + classesStr + "\r\n")
        detailResults = []
        counter = 0
        for idx, (batch, classes, paths) in enumerate(
                unityData):  # return a list of inputs, classes, and path based on the batch parameter of the dataloader
            #print("loop " + str(loopIndex))
            print("tested inputs " + str(totalInputs), end="\r")
            loopIndex = loopIndex + 1
            totalInputs += len(batch)
            batch, classes = Variable(batch), Variable(classes)
            if torch.cuda.is_available():
                batch, classes = batch.cuda(), classes.cuda()
            scores = model(batch)
            scores = scores.detach()
            pred = scores.data.max(1)[1]
            correctPredictedCount += pred.eq(classes.data).cpu().sum()
            for i in range(len(batch)):
                if (classes.data[i].eq(pred[i])):
                    outcome = "Correct"
                else:
                    outcome = "Wrong"
                    counter = counter + 1
                    # print("Number of erronous images " + str(counter))
                strExpected = unityData.dataset.classes[classes[i]]
                strPred = unityData.dataset.classes[pred[i].item()]
                scoreStr = ','.join([str(score) for score in scores[i].data.tolist()])
                if saveFlag:
                    outFile.writelines(paths[i] + "," + outcome + "," + strExpected + "," + strPred + "," + scoreStr[
                                                                                                            1:len(
                                                                                                                scoreStr) - 2] + "\r\n")
        print("Predicted {} out of {} correctly".format(correctPredictedCount, totalInputs))
        print("The average accuracy is: {} %".format(100.0 * correctPredictedCount / (float(totalInputs))))
        print("Total erronous" + str(counter))
        if saveFlag:
            outFile.close()
        return 100.0 * correctPredictedCount / (float(totalInputs)), detailResults


def ieeError(predict_cpu, labels, area, threshold):
    predict_xy = DataSupply.transfer_target(predict_cpu)
    error = []
    for i in range(0, len(predict_cpu)):
        avgDist, maxDist = ieeExtractParts(predict_xy[i], labels.numpy()[i], area)
        if not ((avgDist < 4) and (maxDist < 8)):
            error.append(True)
        else:
            error.append(False)
    return error


def testModelForImg(model, imgClass, trainImage, caseFile):
    if caseFile["datasetName"] == "FLD":
        npyFile = np.load(trainImage.split(".png")[0] + ".npy")
        inputs = npyFile.item()["data"]
        cp_labels = npyFile.item()["label"]
        labels_gt = cp_labels["kps"]
        labels_msk = np.ones(labels_gt.numpy().shape)
        labels_msk[labels_gt.numpy() <= 1e-5] = 0
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        predict = model(inputs)
        predict_cpu = predict.cpu()
        predict_cpu = predict_cpu.detach().numpy()
        predict_xy = DataSupply.transfer_target(predict_cpu)

        diff = np.square(labels_gt.numpy() - predict_xy)
        sum_diff = np.sqrt(diff[:, :, 0] + diff[:, :, 1])
        #avg = sum_diff.mean()

        # print(idx, ": INFO: mean pixel error: ", round(avg,2), " pixels")
        #worst = []
        wlabel = []
        #inputs_cpu = inputs.cpu()
        #inputs_cpu = inputs_cpu.detach().numpy()
        #num_sample = inputs_cpu.shape[0]
        #img = inputs_cpu[0] * 255.
        #img = img[0, :]
        #img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        #max_error = np.max(sum_diff[0])
        avg_error = np.sum(sum_diff[0]) / len(sum_diff[0])
        worst_KP = 0
        label = 0
        worst_label = 0
        for KP in sum_diff[0]:
            if KP > worst_KP:
                worst_KP = KP
                worst_label = label
            label += 1
        wlabel.append(worst_label)
        if avg_error > 4:
            return False, worst_label
        else:
            return True, worst_label

    else:
        dataTransformer = setupTransformer(caseFile["datasetName"])
        transformedData = PathImageFolder(root=caseFile["improveDataPath"], transform=dataTransformer)
        trainDataSet = torch.utils.data.DataLoader(transformedData, batch_size=caseFile["batchSize"], shuffle=True,
                                                   num_workers=caseFile["workersCount"])
        expectedClassID = trainDataSet.dataset.classes.index(imgClass)
        transformer = setupTransformer(caseFile["datasetName"])
        image = Image.open(trainImage)
        imageTensor = transformer(image).float()
        imageTensor = imageTensor.unsqueeze_(0)
        imageTensor = Variable(imageTensor, requires_grad=False)
        imageTensor.detach()
        if torch.cuda.is_available():
            model = model.cuda()
            imageTensor = imageTensor.cuda()
        scores = model(imageTensor)
        scores = scores.detach()
        pred = scores.data.max(1)[1].item()
        if (expectedClassID == pred):
            return True, trainDataSet.dataset.classes[pred]

        else:
            return False, trainDataSet.dataset.classes[pred]


class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(PathImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def testDNN(datasetName, dnn, fileName, filePath, labelPath):
    fileName = str(fileName).split(".")[0]
    json_fn = join(labelPath, fileName + ".json")
    if (datasetName == 'ASL'):
        target = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'DEL', 'NOT', 'SPC']
    if (datasetName == 'GD'):
        target = ['BottomCenter', 'BottomLeft', 'BottomRight', 'MiddleLeft', 'MiddleRight', 'TopCenter', 'TopLeft',
                  'TopRight']

    if (datasetName == 'OC'):
        target = ['Closed', 'Opened']

    # imageName = os.basename(json_fn).replace(".json", ".jpg")
    imgPath = join(filePath, fileName + ".jpg")
    img = cv2.imread(imgPath)
    data_file = open(json_fn)
    data = json.load(data_file)
    look_vec = list(eval(data['eye_details']['look_vec']))
    ldmks_iris = process_json_list(data['iris_2d'], img)
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
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    ldmk1 = ldmks_interior_margin[4]
    ldmk2 = ldmks_interior_margin[12]
    x1 = int(ldmk1[0])
    y1 = int(ldmk1[1])
    x2 = int(ldmk2[0])
    y2 = int(ldmk2[1])

    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist = int(dist)
    vector = np.array(look_vec[:2]) * 80
    milieu_x = getMiddelX(data, img)
    angle, point_A, point_B, point_C = computeAngle(data, img)
    dist_x = getDistBetweenTwoPoints(point_A, milieu_x)

    # angle, milieu_x, milieu_y, intersection, dist_x, dist_y = labelimages.executePiplineForInformation(json_fn)
    if (datasetName == 'GD'):
        if angle >= 0 and angle < 22.5:
            classe = "MiddleLeft"
        if angle > 22.5 and angle < 67.5:
            classe = "TopLeft"
        if angle > 67.5 and angle < 112.5:
            classe = "TopCenter"
        if angle > 112.5 and angle < 157.5:
            classe = "TopRight"
        if angle > 157.5 and angle < 202.5:
            classe = "MiddleRight"
        if angle > 202.5 and angle < 247.5:
            classe = "BottomRight"
        if angle > 247.5 and angle < 292.5:
            classe = "BottomCenter"
        if angle > 292.5 and angle < 337.5:
            classe = "BottomLeft"
        if angle >= 337.5:
            classe = "MiddleLeft"

    if (datasetName == 'OC'):
        if (dist < 20):
            classe = 'Closed'
        else:
            classe = 'Opened'
    image = Image.open(imgPath)
    data_transform = setupTransformer(datasetName)

    image_tensor = data_transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    dnn.eval()
    input = Variable(image_tensor)
    if torch.cuda.is_available():
        input = input.cuda()
    output = dnn.forward(input)
    result = target[output.argmax()]
    if result == classe:
        resStr = 'C'
    else:
        resStr = 'M'
    return resStr


def computeAngle(data, img):
    ldmks_iris = process_json_list(data['iris_2d'], img)
    look_vec = list(eval(data['eye_details']['look_vec']))
    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    # print(look_vec)
    look_vec[1] = -look_vec[1]
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))

    # horizon
    # cv2.line(img, point_A, point_B, (0, 0, 0), 3)
    # cv2.line(img, point_A, point_B, (0, 255, 255), 2)
    # where the eye look
    # cv2.line(img, point_A, point_C, (0, 0, 0), 3)
    # cv2.line(img, point_A, point_C, (0, 255, 255), 2)

    angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
                                                                                      point_B[1] - point_A[1]);

    # print( "1st Angle" )
    # print( angle )

    angle = (angle * 180) / math.pi

    # print( "2nd Angle" )
    # print(angle)

    while (angle < 0):
        angle = angle + 360

    # print( "3rd Angle" )
    # print(angle)

    return angle, point_A, point_B, point_C


def getDistBetweenTwoPoints(point_A, milieu_x):
    return math.sqrt((point_A[0] - milieu_x[0]) * (point_A[0] - milieu_x[0]) + (point_A[1] - milieu_x[1]) * (
            point_A[1] - milieu_x[1]))


def process_json_list(json_list, img):
    ldmks = [eval(s) for s in json_list]
    return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])


def getMiddelX(data, img):
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    return milieu(int(ldmks_interior_margin[0][0]), int(ldmks_interior_margin[0][1]),
                  int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0]),
                  int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1]))


def milieu(x1, y1, x2, y2):
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return [x, y]
# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
