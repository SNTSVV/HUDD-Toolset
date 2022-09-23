import paramsModule as pM
from imports import np, os, math, sc, stat, pd, torch, Workbook, dataframe_to_rows, join, basename


def UnityRQ1(caseFile):
    normalizedValues = [0.06, 0.11, 0.19, 0.32, 0.32,0.43,0.44,0.47,0.50,0.51,0.56,0.56,0.69,0.74,0.82,0.88,0.94,0.94]
    #normalizedValues = [0.06,0.07,0.19,0.26,0.32,0.44,0.48,0.56,0.58,0.60,0.69,0.74,0.82,0.93,0.94]
    DataSetsPath = caseFile["DataSetsPath"]
    datasetName = caseFile["datasetName"]
    xlPath = join(caseFile["filesPath"], "Layer0HMDistance.xlsx")
    outExcelPath = join(caseFile["outputPath"], caseFile["datasetName"] + "_RQ1.xlsx")
    clustersPath = join(caseFile["outputPath"], "ClusterAnalysis_" + caseFile["clustMode"])
    layers = caseFile["selectedLayer"]
    ClusterOf = {}
    N_clusters = len(normalizedValues)
    for C in range(0, N_clusters):
        ClusterOf[C] = C
    clusterCOUNTER = N_clusters
    for D in range(0, 11):
        for S in range(1, N_clusters):
            for P in range(0, N_clusters):
                if (P+S) >= N_clusters:
                    break
                merge = False
                if float(normalizedValues[P + S] - normalizedValues[P]) <= float(D/100):
                    merge = True
                if P > 0:
                    if ClusterOf[P-1] == ClusterOf[P]:
                        merge = False
                if P + S < (N_clusters-1):
                    if ClusterOf[P+S] == ClusterOf[P+S+1]:
                        merge = False
                if merge:
                    clusterCOUNTER += 1
                    for Element in range(P, P+S+1):
                        #print(Element, clusterCOUNTER)
                        ClusterOf[Element] = clusterCOUNTER
    print(ClusterOf)
    Train_dir = join(DataSetsPath, "TrainingSet")
    Train_json = join(DataSetsPath, "TrainingSet_json")
    Test_dir = join(DataSetsPath, "TestSet")
    Test_json = join(DataSetsPath, "TestSet_json")
    Set = list()
    xlFile = pd.read_excel(xlPath)
    i = 0
    for col in xlFile.columns:
        if i == 0:
            i = 1
        else:
            Set.append(col)
    labelList = list()
    AvgICDList = list()
    SilList = list()
    newAvgICDList = list()
    for layer in layers:
        clusters = torch.load(join(str(clustersPath), layer + ".pt"))
        labelListed = clusters['label list']
        labelList.append([(i) for i in labelListed])
        AvgICDList.append(clusters['avgLayer'])
        newAvgICDList.append(clusters['avgLayer'])
        SilList.append(clusters['silhouette'])
    excel(labelList, AvgICDList, SilList, newAvgICDList, Set, Train_dir, Test_dir, Train_json, Test_json,
          datasetName, outExcelPath, layers)


def excel(cc, AvgICDList, SilList, newAvgICDList, FullTestSet, Train_dir, Test_dir, Train_json, Test_json, mode, clustexcel, layers):
    arr = [0] * len(cc)
    wb = Workbook()
    wb.save(clustexcel)
    red = [0] * (11)
    discarded = [0] * (11)
    i = 0
    for layer in layers:
        df1, df2, df3, arr[i], df4, discarded[i], red[i] = pM.excelsheet(cc[i], FullTestSet, Train_dir, Test_dir, Train_json, Test_json,
                                                                   mode)
        sheet = wb.create_sheet(str(layer), 0)
        for x in dataframe_to_rows(df1):
            sheet.append(x)
        sheet2 = wb.create_sheet('Summary_' + str(layer), 0)
        for x in dataframe_to_rows(df2):
            sheet2.append(x)
        sheet3 = wb.create_sheet('Thresholds' + str(layer), 0)
        z = 0
        for x in dataframe_to_rows(df3):

            if (z != 1):
                sheet3.append(x)

            z = z + 1
        i = i + 1
    labels = ["Layer#", "Cluster#", "AvgDisp_Var%_All", "AvgDisp_Var%_1", "AvgDisp_Var%_3", "AvgDisp_Var%_4", "AvgDisp_Var%_Highest50%+", "Pass#_Angle",
              "Pass#_HHeadpose", "Pass#_VHeadpose", "Pass#_StrangeDist", "Pass#_Dist", "LayerScore1", "LayerScore2",
              "NumClusters_Had_Below50", "NumClusters_Had_Below50_CrossingBoundary", "%ClstBelow50", "%ClstBelowCross"]

    i = 0
    for layer in layers:
        arr[i] = [layer] + arr[i]
        i = i + 1
    col = [0] * len(labels)
    for i in range(0, len(col)):
        col[i] = [0] * len(arr)
    for i in range(0, len(labels)):
        for j in range(0, len(arr)):
            col[i][j] = arr[j][i]
        df4[labels[i]] = col[i]
    ChosenLayers = ['']*4
    ChosenLayers[0] = bestlayer(red, min(red), layers)
    ChosenLayers[1] = bestlayer(AvgICDList, min(AvgICDList), layers)
    ChosenLayers[2] = bestlayer(SilList, max(SilList), layers)
    ChosenLayers[3] = bestlayer(newAvgICDList, min(newAvgICDList), layers)
    df4["# Discarded Clusters"] = discarded
    df4["AvgPass_Angle"] = [i / j for i, j in zip(col[6], col[1])]
    df4["AvgPass_HHeadpose"] = [i / j for i, j in zip(col[7], col[1])]
    df4["AvgPass_VHeadpose"] = [i / j for i, j in zip(col[8], col[1])]
    df4["AvgPass_StrangeDist"] = [i / j for i, j in zip(col[9], col[1])]
    df4["AvgPass_Dist"] = [i / j for i, j in zip(col[10], col[1])]
    df4["Avg Reduction in Variance [VarP[clusters]/(n_clusters-discarded)]"] = red
    #df4["avg1"]
    #df4["avg3"]
    #df4["avg4"]
    #df4["avgall"]
    #df4["avgallbelow50"]
    #df4["avg3below50"]
    df4["AvgDist_Per_DataPoint"] = AvgICDList
    df4["Silhouette"] = SilList
    df4["NewAvgICD"] = newAvgICDList
    df4["AvgRed/AvgICD/Silh/newAvgICD"] = pd.Series(ChosenLayers)
    sheet4 = wb.create_sheet("Summary", 0)
    i = 0
    for x in dataframe_to_rows(df4):
        if(i != 1):
            sheet4.append(x)
        i = i+1
    wb.save(clustexcel)


def bestlayer(avgList, minAvg, layers):
    minIndex = [i for i, j in enumerate(avgList) if j == minAvg]
    minIndex = minIndex[0]
    minLayer = layers[minIndex]
    return minLayer


def getDetails(VarRedC, DictList):
    #print(100 - (VarRedC * 100), sum(DictList)/len(DictList), min(DictList), max(DictList), stat.variance(DictList),
    #      stat.median(DictList), stat.stdev(DictList), np.percentile(DictList, 25), np.percentile(DictList, 75))
    #print(DictList)
    return [100 - (VarRedC * 100), sum(DictList)/len(DictList), min(DictList), max(DictList), stat.variance(DictList),
            stat.median(DictList), stat.stdev(DictList), np.percentile(DictList, 25), np.percentile(DictList, 75)]


def IEERQ1(caseFile):
        faceSubset = caseFile["faceSubset"] # Face component
        testNPY = caseFile["testDataNpy"] # testdataset npy file
        trainNPY = caseFile["trainDataNpy"] # traindataset npy
        trainCSV = caseFile["trainCSV"] # csv file - train
        trainCSV = None # csv file - train
        testCSV = caseFile["testCSV"] # csv file - test
        layers = caseFile["layers"] ##
        Alex = caseFile["Alex"] ## HPD
        FLD = caseFile["FLD"] ## FLD
        clustMode = caseFile["clustMode"]
        outputPath = caseFile["outputPath"]
        dataset1 = np.load(trainNPY, allow_pickle=True)
        dataset1 = dataset1.item()
        x_data = dataset1["label"]
        x_config = dataset1["config"]
        clustersData = {}
        if not Alex:
            clustexcel = join(outputPath, faceSubset + "_RQ1.xlsx")
        else:
            clustexcel = join(outputPath, "RQ1.xlsx")
        writer = pd.ExcelWriter(clustexcel, engine='xlsxwriter')
        arr = list()
        counter = 0
        for z in x_config[0]:
            print(z)
            print(x_config[0][z])
            if isinstance(x_config[0][z], str):
                #print(x_config[0][z])
                counter = counter
            elif isinstance(x_config[0][z], float):
                #print(x_config[0][z])
                arr.append(z)
                counter += 1
            elif len(x_config[0][z]) == 1:
                arr.append(z)
                #print(x_config[0][z])
                counter += 1
            else:
                i = 0
                for h in x_config[0][z]:
                    arr.append(z + str(i))
                    #print(z + str(i), h)
                    i += 1
                    counter += 1

        dataset2 = np.load(testNPY, allow_pickle=True)
        dataset2 = dataset2.item()
        y_data = dataset2["label"]
        y_config = dataset2["config"]
        totalClustersList = list()
        PercOfRedClustersList = list()
        HighestVarRedOverClustersList = list()
        AvgVarDispList = list()
        layers = [caseFile["selectedLayer"]] # ["clustersFile"] -- clustersFile.pt   # {'clusters': 1: 'members': X,Y, 2: 'members':}
        layers = ["Layer15"] # ["clustersFile"] -- clustersFile.pt   # {'clusters': 1: 'members': X,Y, 2: 'members':}
        for layerX in layers:
            #layerClust = join(caseFile["outputPath"], caseFile["RCC"], "ClusterAnalysis_" + str(clustMode), layerX + ".pt")
            layerClust = join(caseFile["outputPath"], caseFile["RCC"], "ClusterAnalysis_" + str(clustMode), layerX + ".pt")
            clsData = torch.load(layerClust, map_location=torch.device('cpu'))
            clusterDictX = {}
            clusterDictY = {}
            clusterDictP = {}
            clusterDictC = {}
            allDictX = {}
            allDictY = {}
            allDictP = {}
            allDictC = {}
            VarRedXDict = {}
            VarRedYDict = {}
            VarRedPDict = {}
            VarRedCDict = {}
            for i in range(0, counter):
                allDictC[i] = []
            for i in range(0, 27):
                allDictX[i] = []
                allDictY[i] = []
                allDictP[i] = []
            for clusterID in clsData['clusters']:
                if len(clsData['clusters'][clusterID]['members']) > 1:
                    groupMembers = clsData['clusters'][clusterID]['members']
                    clusterDictX[clusterID] = {}
                    clusterDictY[clusterID] = {}
                    clusterDictP[clusterID] = {}
                    clusterDictC[clusterID] = {}
                    for i in range(0, 27):
                        clusterDictX[clusterID][i] = []
                        clusterDictY[clusterID][i] = []
                        clusterDictP[clusterID][i] = []
                    for i in range(0, counter):
                        clusterDictC[clusterID][i] = []
                    if trainCSV is not None:
                        imageList = pd.read_csv(trainCSV)
                        for index, row in imageList.iterrows():
                            if row["result"] == "Wrong":
                                fileName = None
                                dataIndex = None
                                if not Alex:
                                    if row["worst_component"] == faceSubset:
                                        fileName = "Train_" + row["image"].split(".png")[0]
                                        dataIndex = index
                                else:
                                    fileName = "Train_" + str(basename(row["image"])).split(".png")[0] + "_" + row["expected"]
                                    dataIndex = int(str(basename(row["image"])).split(".png")[0]) - 1
                                if groupMembers.count(fileName)>0:
                                    labels = x_data[dataIndex]
                                    x = 0
                                    for KP in labels:
                                        clusterDictX[clusterID][x].append(KP[0])
                                        allDictX[x].append(KP[0])
                                        clusterDictY[clusterID][x].append(KP[1])
                                        allDictY[x].append(KP[1])
                                        Z = KP[0]**2 + KP[1]**2
                                        clusterDictP[clusterID][x].append(math.sqrt(Z))
                                        allDictP[x].append(math.sqrt(Z))
                                        x += 1
                                    x = 0
                                    params = x_config[dataIndex]
                                    for param in params:
                                        if isinstance(params[param], str):
                                            #clusterDictC[clusterID][x].append(params[param])
                                            #allDictC[x].append(params[param])
                                            x = x
                                        elif isinstance(params[param], float):
                                            clusterDictC[clusterID][x].append(float(params[param]))
                                            allDictC[x].append(float(params[param]))
                                            x += 1
                                        elif len(params[param]) == 1:
                                            clusterDictC[clusterID][x].append(float(params[param]))
                                            allDictC[x].append(float(params[param]))
                                            x += 1
                                        else:
                                            i = 0
                                            for parami in params[param]:
                                                clusterDictC[clusterID][x].append(float(parami))
                                                allDictC[x].append(float(parami))
                                                #print(param + str(i), parami)
                                                i += 1
                                                x += 1
                    imageList = pd.read_csv(testCSV)
                    for index, row in imageList.iterrows():
                        if row["result"] == "Wrong":
                            fileName = None
                            dataIndex = None
                            if not Alex:
                                if row["worst_component"] == faceSubset:
                                    fileName = "Test_" + row["image"].split(".png")[0]
                                    dataIndex = index
                            else:
                                fileName = "Test_" + str(basename(row["image"])).split(".png")[0] + "_" + row["expected"]
                                dataIndex = int(str(basename(row["image"])).split(".png")[0]) - 1
                            if groupMembers.count(fileName)>0:
                                labels = y_data[dataIndex]
                                x = 0
                                for KP in labels:
                                    clusterDictX[clusterID][x].append(KP[0])
                                    allDictX[x].append(KP[0])
                                    clusterDictY[clusterID][x].append(KP[1])
                                    allDictY[x].append(KP[1])
                                    Z = KP[0]**2 + KP[1]**2
                                    clusterDictP[clusterID][x].append(math.sqrt(Z))
                                    allDictP[x].append(math.sqrt(Z))
                                    x += 1
                                params = y_config[dataIndex]
                                x = 0
                                for param in params:
                                    if isinstance(params[param], str):
                                        #clusterDictC[clusterID][x].append(params[param])
                                        #allDictC[x].append(params[param])
                                        x = x
                                    elif isinstance(params[param], float):
                                        clusterDictC[clusterID][x].append(float(params[param]))
                                        allDictC[x].append(float(params[param]))
                                        x += 1
                                    elif len(params[param]) == 1:
                                        clusterDictC[clusterID][x].append(float(params[param]))
                                        allDictC[x].append(float(params[param]))
                                        x += 1
                                    else:
                                        i = 0
                                        for parami in params[param]:
                                            clusterDictC[clusterID][x].append(float(parami))
                                            allDictC[x].append(float(parami))
                                            #print(param + str(i), parami)
                                            i += 1
                                            x += 1
            print("Layer: ", layerX)
            AvgVarDictX = {}
            AvgVarDictY = {}
            AvgVarDictP = {}
            AvgVarDictC = {}
            df3 = pd.DataFrame()
            for KP in range(0, 27):
                AvgVarDictX[KP] = []
                AvgVarDictY[KP] = []
                AvgVarDictP[KP] = []
            for j in range(0, counter):
                AvgVarDictC[j] = []
            RedClusters = 0
            totalClusters = 0
            HighestVarRedOverClusters = list()
            for clusterID in clsData['clusters']:
                print("CLUSTER:", clusterID)
                clustersData[clusterID] = {}
                if len(clsData['clusters'][clusterID]['members']) > 1:
                    totalClusters += 1
                    HighVarRed = list()
                    check = 0
                    VarRedXDict[clusterID] = {}
                    VarRedYDict[clusterID] = {}
                    VarRedPDict[clusterID] = {}
                    VarRedCDict[clusterID] = {}
                    highestX = 0.5
                    highestY = 0.5
                    highestP = 0.5
                    highestC = 0.5
                    #for KP in range(0, 27):
                        #if (len(clusterDictX[clusterID][KP]) > 1) and (len(allDictX[KP]) > 1):
                            #VarRedX = stat.variance(clusterDictX[clusterID][KP])/stat.variance(allDictX[KP])
                            #VarRedY = stat.variance(clusterDictY[clusterID][KP])/stat.variance(allDictY[KP])
                            #VarRedP = stat.variance(clusterDictP[clusterID][KP])/stat.variance(allDictP[KP])
                            #AvgVarDictX[KP].append(VarRedX)
                            #AvgVarDictY[KP].append(VarRedY)
                            #AvgVarDictP[KP].append(VarRedP)
                            #if VarRedX < highestX:
                            #    VarRedXDict[clusterID]["KP"+str(KP)] = getDetails(VarRedX, clusterDictX[clusterID][KP])
                            #if VarRedP < highestY:
                            #    VarRedYDict[clusterID]["KP"+str(KP)] = getDetails(VarRedY, clusterDictY[clusterID][KP])
                            #if VarRedP < highestP:
                            #    VarRedPDict[clusterID]["KP"+str(KP)] = getDetails(VarRedP, clusterDictP[clusterID][KP])
                            #HighVarRed.append(VarRedX)
                            #HighVarRed.append(VarRedY)
                            #HighVarRed.append(VarRedP)
                    VarRedCDict[clusterID] = {}
                    for j in range(0, counter):
                        if (len(clusterDictC[clusterID][j]) > 1) and (len(allDictC[j]) > 1):
                            if (stat.variance(allDictC[j]) == 0.0):
                                VarRedC = 1.0
                            else:
                                VarRedC = stat.variance(clusterDictC[clusterID][j])/stat.variance(allDictC[j])
                            AvgVarDictC[j].append(VarRedC)
                            HighVarRed.append(VarRedC)
                            #HighVarRed.append(VarRedC)
                            #HighVarRed.append(VarRedC)
                            #if VarRedC < highestC:
                            #    VarRedCDict[clusterID][str(arr[j])] = getDetails(VarRedC, clusterDictC[clusterID][j])
                            if VarRedC < 0.5:
                                #if arr[j].startswith("head"):
                                #print(arr[j])
                                #print("Min in Cluster", min(clusterDictC[clusterID][j]))
                                #print("Max in Cluster", max(clusterDictC[clusterID][j]))
                                #print("Avg in Cluster", sum(clusterDictC[clusterID][j])/len(clusterDictC[clusterID][j]))
                                #print("Min", min(allDictC[j]))
                                #print("Max", max(allDictC[j]))
                                #print("Avg", sum(allDictC[j])/len(allDictC[j]))
                                    #print(allDictC[j])
                                VarRedCDict[clusterID][str(arr[j])] = getDetails(VarRedC, clusterDictC[clusterID][j])
                                #print("1st Quartile:", VarRedCDict[clusterID][str(arr[j])][len(VarRedCDict[clusterID][str(arr[j])])-1])
                                #print("3rd Quartile:", VarRedCDict[clusterID][str(arr[j])][len(VarRedCDict[clusterID][str(arr[j])])-2])
                    #number of clusters had high variance reduction below 50% and unsafe
                    NumParams = 1
                    print("num params", NumParams)
                    #if FLD == 1:
                    #    NumParams = 3
                    #elif FLD == 2:
                    #    NumParams = 4
                    #elif FLD == 0:
                    #    NumParams = 4
                    check = 0
                    check1 = 0
                    check2 = 0
                    check3 = 0
                    check4 = 0
                    check5 = 0
                    check6 = 0
                    check7 = 0
                    check8 = 0
                    check9 = 0
                    check10 = 0
                    for i in range(0, NumParams):
                        print("Highest Var Red", 100*(1-min(HighVarRed)))
                        HighestVarRed = min(HighVarRed)
                        HighestVarRedOverClusters.append(HighestVarRed)
                        if HighestVarRed < 0.10:
                            check1 += 1
                        if HighestVarRed < 0.20:
                            check2 += 1
                        if HighestVarRed < 0.30:
                            check3 += 1
                        if HighestVarRed < 0.40:
                            check4 += 1
                        print(HighestVarRed)
                        if HighestVarRed < 0.50:
                            check5 += 1
                            check += 1
                            print("here")
                        if HighestVarRed < 0.60:
                            check6 += 1
                        if HighestVarRed < 0.70:
                            check7 += 1
                        if HighestVarRed < 0.80:
                            check8 += 1
                        if HighestVarRed < 0.90:
                            check9 += 1
                        if HighestVarRed < 1:
                            check10 += 1
                        HighVarRed.remove(HighestVarRed)
                    #print(check1, check2, check3, check4, check5, check6, check7, check8, check9, check10)
                    if check >= NumParams:
                        RedClusters += 1
                    dfList = list()
                    for x in VarRedXDict[clusterID]:
                        keyName = x+str("_X")
                        if keyName not in clustersData[clusterID]:
                            clustersData[clusterID][keyName] = list()
                        dfList.append(keyName)
                        y = 0
                        for val in VarRedXDict[clusterID][x]:
                            dfList.append(val)
                            y += 1
                            if y > len(VarRedXDict[clusterID][x])-2:
                                clustersData[clusterID][keyName].append(val)
                        #dfList.append(VarRedXDict[clusterID][x])
                        #df3 = df3.append({clusterID: x+str("_X")}, ignore_index=True)
                        #df3 = df3.append({clusterID: VarRedXDict[clusterID][x]}, ignore_index=True)
                    for x in VarRedYDict[clusterID]:
                        keyName = x+str("_Y")
                        if keyName not in clustersData[clusterID]:
                            clustersData[clusterID][keyName] = list()
                        dfList.append(keyName)
                        y = 0
                        for val in VarRedYDict[clusterID][x]:
                            dfList.append(val)
                            y += 1
                            if y > len(VarRedYDict[clusterID][x])-2:
                                clustersData[clusterID][keyName].append(val)
                        #dfList.append(VarRedYDict[clusterID][x])
                        #df3 = df3.append({clusterID: x+str("_Y")}, ignore_index=True)
                        #df3 = df3.append({clusterID: VarRedYDict[clusterID][x]}, ignore_index=True)
                    for x in VarRedPDict[clusterID]:
                        keyName = x+str("_P")
                        if keyName not in clustersData[clusterID]:
                            clustersData[clusterID][keyName] = list()
                        dfList.append(keyName)
                        y = 0
                        for val in VarRedPDict[clusterID][x]:
                            dfList.append(val)
                            y += 1
                            if y > len(VarRedPDict[clusterID][x])-2:
                                clustersData[clusterID][keyName].append(val)
                        #dfList.append(VarRedPDict[clusterID][x])
                        #df3 = df3.append({clusterID: x+str("_P")}, ignore_index=True)
                        #df3 = df3.append({clusterID: VarRedPDict[clusterID][x]}, ignore_index=True)
                    for x in VarRedCDict[clusterID]:
                        keyName = x+str("_C")
                        if keyName not in clustersData[clusterID]:
                            clustersData[clusterID][keyName] = list()
                        dfList.append(keyName)
                        y = 0
                        for val in VarRedCDict[clusterID][x]:
                            dfList.append(val)
                            y += 1
                            if y > len(VarRedCDict[clusterID][x])-2:
                                clustersData[clusterID][keyName].append(val)
                        clustersData[clusterID][keyName].append(min(allDictC))
                        clustersData[clusterID][keyName].append(max(allDictC))
                        #dfList.append(VarRedCDict[clusterID][x])
                        #df3 = df3.append({clusterID: x}, ignore_index=True)
                        #df3 = df3.append({clusterID: VarRedCDict[clusterID][x]}, ignore_index=True)
                    #df3 = df3.dropna()
                    df3[clusterID] = pd.Series(dfList)
                    df3.to_excel(writer, sheet_name=str(layerX))
            PercOfRedClusters = (RedClusters/totalClusters)*100.0
            AvgVarDisp = 0
            counterX = 0
            print("reduced", RedClusters)
            print("Perc%%%%", PercOfRedClusters)
            #for KP in range(0, 27):
            #    Val1 = sum(AvgVarDictX[KP])/len(AvgVarDictX[KP])
            #    Val2 = sum(AvgVarDictY[KP])/len(AvgVarDictY[KP])
            #    Val3 = sum(AvgVarDictP[KP])/len(AvgVarDictP[KP])
            #    AvgVarDisp += Val1 + Val2 + Val3
            #    counterX += 3
            for j in range(0, counter):
                AvgVarDisp += sum(AvgVarDictC[j])/len(AvgVarDictC[j])
                counterX += 1
            print("Total Clusters,", "% of clusters,", "Avg var of highest params,", "Avg var of all params")
            print(str(totalClusters), "%.2f" % PercOfRedClusters,
                  "%.2f" % float(100-((sum(HighestVarRedOverClusters)/len(HighestVarRedOverClusters))*100.0)),
                  "%.2f" % float(100-((AvgVarDisp/counterX)*100.0)))
            totalClustersList.append(totalClusters)
            PercOfRedClustersList.append(PercOfRedClusters)
            HighestVarRedOverClustersList.append(float(100-((sum(HighestVarRedOverClusters)/len(HighestVarRedOverClusters))*100.0)))
            AvgVarDispList.append(float(100-((AvgVarDisp/counterX)*100.0)))
            #print("Average Variance reduction of all params over all clusters:", 100-((AvgVarDisp/counter)*100.0))
            #print("Percentage of clusters with at least 3 param having >50% reduction:", PercOfRedClusters, "%")
            #print("Average variance of highest param over all clusters:", 100-(((sum(HighestVarRedOverClusters)/len(HighestVarRedOverClusters))*100.0)))
            #print("Total Non-single Clusters:", totalClusters)
            #print("Median-AD Reduction:", AvgMadDisp/counter)
            #print("Mean-AD Reduction:", AvgMedDisp/counter)
        labelList = list()
        AvgICDList = list()
        SilList = list()
        newAvgICDList = list()
        for layer in layers:
            layerClust = join(outputPath, caseFile["RCC"], "ClusterAnalysis_" + str(clustMode), layer + ".pt")
            clsData = torch.load(layerClust,map_location=torch.device('cpu'))
            labelListed = clsData['label list']
            labelList.append([(i) for i in labelListed])
            AvgICDList.append(clsData['avgLayer'])
            newAvgICDList.append(clsData['avgLayer'])
            SilList.append(clsData['silhouette'])
        wb = Workbook()
        #wb.save(clustexcel)
        red = [0] * (11)
        discarded = [0] * (11)
        df4 = pd.DataFrame()
        AvgICDList[0] = 50000000000
        SilList[0] = -50000000000
        ChosenLayers = [''] * 3
        ChosenLayers[0] = bestlayer(red, min(red), layers)
        ChosenLayers[1] = bestlayer(AvgICDList, min(AvgICDList), layers)
        ChosenLayers[2] = bestlayer(SilList, max(SilList), layers)
        df4["layers"] = layers
        df4["total Clusters"] = totalClustersList
        df4["percentage of reduced Clusters"] = PercOfRedClustersList
        df4["Avg var reduction of highest 3 params"] = HighestVarRedOverClustersList
        df4["Avg var reduction of all params"] = AvgVarDispList
        df4["Silhouette"] = SilList
        df4["avgLayerICD"] = AvgICDList
        df4["Silh"] = pd.Series(bestlayer(SilList, max(SilList), layers))
        df4["AvgICD"] = pd.Series(bestlayer(AvgICDList, min(AvgICDList), layers))
        #df4["AvgRedOf1"] = pd.Series(bestlayer(HighestVarRedOverClustersList, max(HighestVarRedOverClustersList), layers))
        df4["AvgRedOf3"] = pd.Series(bestlayer(HighestVarRedOverClustersList, max(HighestVarRedOverClustersList), layers))
        #df4["AvgRedOf3below50"] = pd.Series(bestlayer(HighestVarRedOverClustersList, max(HighestVarRedOverClustersList), layers))
        #df4["AvgRedOf4"] = pd.Series(bestlayer(HighestVarRedOverClustersList, max(HighestVarRedOverClustersList), layers))
        df4["AvgRedOfAll"] = pd.Series(bestlayer(AvgVarDispList, max(AvgVarDispList), layers))
        #df4["AvgRedOfAllbelow50"] = pd.Series(bestlayer(AvgVarDispList, max(AvgVarDispList), layers))
        df4.to_excel(writer, sheet_name="Summary")
        #sheet4 = wb.create_sheet("Summary", 0)
        #i = 0
        #for x in dataframe_to_rows(df4):
        #    if (i != 1):
        #        sheet4.append(x)
        #    i = i + 1
        #wb.save(clustexcel)
        writer.close()
        return clustersData



def thresholds(v, th):
    if v < th:
        print(th)
        return True