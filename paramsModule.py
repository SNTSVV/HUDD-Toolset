#
# Copyright (c) University of Luxembourg 2019-2020.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
#
import processImages as labelImages
from imports import os, pd, sc, stat, np, math, json, cv2, join, torch, basename, dirname

def excelsheet(cc, unsafe_LR, Train_dir, Test_dir, Train_json, Test_json, mode):
    DNN_Result = [0] * len(unsafe_LR)
    Angle = [0] * len(unsafe_LR)
    Distance = [0] * len(unsafe_LR)
    OC = [0] * len(unsafe_LR)
    Pupil = [0] * len(unsafe_LR)
    Iris = [0] * len(unsafe_LR)
    Skybox_expo = [0] * len(unsafe_LR)
    Skybox_rot = [0] * len(unsafe_LR)
    Light = [0] * len(unsafe_LR)
    Ambien = [0] * len(unsafe_LR)
    HeadposeX = [0] * len(unsafe_LR)
    HeadposeY = [0] * len(unsafe_LR)
    GroundTruth = [0] * len(unsafe_LR)
    StrangeDistBot = [0] * len(unsafe_LR)
    StrangeDistTop = [0] * len(unsafe_LR)
    StrangeDetect = [0] * len(unsafe_LR)
    Dist_x = [0] * len(unsafe_LR)
    #print(cc)
    n_param = 13
    labels = cc
    #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_clusters = max(labels)
    #n_noise = list(labels).count(-1)
    print("Collecting clusters data")
    if(mode == 'GD' or mode == 'OC'):
        df1 = pd.DataFrame(columns={"FileName", "Angle", "Distance", "Pupil Size", "Iris Size", "Skybox Exposure",
             "Skybox Rotation", "Light Intenisty", "Ambient Intenisty", "HeadPoseX", "HeadPoseY",
             "AngleNorm", "DistanceNorm", "Pupil Size Norm", "Iris Size Norm", "Skybox Exposure Norm",
             "Skybox Rotation Norm", "Light Intenisty Norm", "Ambient Intenisty Norm", "HeadPoseX Norm", "HeadPoseY Norm", "Open/Closed", "GroundTruth", "DNN_Result", "Best Feature Percentage",
                                    "Second Best Feature Percentage"})
        for i in range(0, len(unsafe_LR)):
            file = unsafe_LR[i]
            #result, layerHM = dnn.classifyOneImage(file, net, orig_dir, new_dir,12,0, mode)
            result = ''
            fileSource = str(file.split("_")[0])
            if fileSource == "Train":
                jsonx = Train_json
                orig_dir = Train_dir
            elif fileSource == "Test":
                jsonx = Test_json
                orig_dir = Test_dir
            fileClass = str(file.split("_")[2])
            fileName = str(file.split("_")[1])
            img = cv2.imread(join(orig_dir,fileClass, fileName + ".jpg"))
            json_fn = join(jsonx, fileName + ".json")
            data_file = open(json_fn)
            data = json.load(data_file)
            look_vec = list(eval(data['eye_details']['look_vec']))
            ldmks_iris = labelimages.process_json_list(data['iris_2d'], img)
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
            ldmks_interior_margin = labelimages.process_json_list(data['interior_margin_2d'], img)
            ldmk1 = ldmks_interior_margin[4]
            ldmk2 = ldmks_interior_margin[12]
            x1 = int(ldmk1[0])
            y1 = int(ldmk1[1])
            x2 = int(ldmk2[0])
            y2 = int(ldmk2[1])

            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dist = int(dist)
            # print(data['lighting_details']['light_intensity'])
            skybox = float(data['lighting_details']['skybox_exposure'])
            skybox_r = float(data['lighting_details']['skybox_rotation'])
            intenisty = float(data['lighting_details']['light_intensity'])
            ambient = float(data['lighting_details']['ambient_intensity'])
            pupil = float(data['eye_details']['pupil_size'])
            iris = float(data['eye_details']['iris_size'])
            head_pose = data['head_pose']
            hp = head_pose
            hp1 = float(hp.split(",")[0].split("(")[1])
            hp2 = float(hp.split(", ")[1].split(",")[0])
            vector = np.array(look_vec[:2]) * 80
            sumvector = math.sqrt((vector[0]**2) + (vector[1]**2))
            milieu_x = labelimages.getMiddelX(data, img)
            angle, point_A, point_B, point_C = labelimages.computeAngle(data, img)
            dist_x = labelimages.getDistBetweenTwoPoints(point_A, milieu_x)

            #angle, milieu_x, milieu_y, intersection, dist_x, dist_y = labelimages.executePiplineForInformation(json_fn)
            if(mode=='GD'):
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
                #if ((337.5 <= angle < 22.5) or ( 157.5 <= angle < 202.5)):
                    # chance to be center center!!
                    #if dist_x <= 25:
                    #    classe = "MiddleCenter"
                #if ((angle == 0) or (angle == 45) or (angle == 90) or (angle == 135) or (angle == 180) or (angle == 225) or (angle == 270) or (angle == 315)):
                #    classe = "MiddleCenter"

                if(look_vec[2] < -0.997 and dist_x <= 29):
                    classe = "MiddleCenter"
                GroundTruth[i] = classe

            if(mode=='OC'):
                if (dist < 20):
                    OC[i] = 'C'
                else:
                    OC[i] = 'O'
                GroundTruth[i] = OC[i]
            #Dist_x[i] = dist_x
            Dist_x[i] = sumvector
            StrangeDistBot[i], StrangeDistTop[i], StrangeDetect[i] = labelimages.detect_strange(json_fn, img, -15)
            DNN_Result[i] = result
            Angle[i] = angle
            Distance[i] = dist
            Pupil[i] = pupil
            Iris[i] = iris
            Skybox_expo[i] = skybox
            Skybox_rot[i] = skybox_r
            Light[i] = intenisty
            Ambien[i] = ambient
            HeadposeX[i] = hp1
            HeadposeY[i] = hp2
    #AngleNorm, DistanceNorm, PupilNorm, IrisNorm, Skybox_expoNorm, Skybox_rotNorm, LightNorm, AmbienNorm, HeadposeXNorm, HeadposeYNorm =
    # do_norm(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, n_param)
    var_A = do_var(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, n_param)
    medad_A = do_medad(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop,Dist_x, n_param)
    meanad_A = do_meanad(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop,Dist_x, n_param)
    #def do_norm():
    #= [(float(i) - min(HeadposeY)) / (max(HeadposeY) - min(HeadposeY)) for i in HeadposeY]
    print("Angle")
    AllM = [0] * len(Angle)
    for i in range(0, len(Angle)):
        AllM[i] = (Angle[i] * (math.pi / 180))
    avar = sc.circvar(AllM, high=2 * math.pi, low=0)
    print(avar)
    print("Distance")
    print(stat.variance(Distance))
    print("Pupil Size")
    print(stat.variance(Pupil))
    print("Iris Size")
    print(stat.variance(Iris))
    print("Skybox Exposure")
    print(stat.variance(Skybox_expo))
    print("Skybox Rotation")
    print(stat.variance(Skybox_rot))
    print("Light Intenisty")
    print(stat.variance(Light))
    print("Ambient")
    print(stat.variance(Ambien))
    print("HeadPoseX")
    print(stat.variance(HeadposeX))
    print("HeadPoseY")
    print(stat.variance(HeadposeY))
    print("StrangeDistBot")
    print(stat.variance(StrangeDistBot))
    print("StrangeDistTop")
    print(stat.variance(StrangeDistTop))
    print("Dist_x")
    print(stat.variance(Dist_x))
    print(Angle[0])
    df2 = pd.DataFrame()
    df4 = pd.DataFrame()
    var_c_clusters = [0] * n_clusters
    mad_c_clusters = [0] * n_clusters
    med_c_clusters = [0] * n_clusters
    avg_c_clusters = [0] * n_clusters
    avganalysis1 = [0] * n_clusters
    avganalysis2 = [0] * n_clusters
    avganalysis3 = [0] * n_clusters
    avganalysis4 = [0] * n_clusters
    avganalysis5 = [0] * n_clusters

    reduced_parameter = [0] * n_clusters
    reduced_param_val = [0] * n_clusters
    sucess1 = 0
    sucess2 = 0
    sucess3 = 0
    clustpass = 0
    discarded = 0
    numpass1 = 0
    numpass2 = 0
    numpass3 = 0
    numpass4 = 0
    numpass5 = 0
    num_below = 0
    num_test = 0
    print("Starting groups analysis")
    print("Number of clusters are " + str(n_clusters))
    for label in range(1, n_clusters+1):
        df1 = pd.DataFrame(columns={"FileName"})
        df3 = pd.DataFrame()
        j = 0
        group_len = list(labels).count(label)
        if(group_len > 1):
            group_indices = [0] * group_len
            Files = [0] * group_len
            GDNN_Result = [0] * group_len
            GAngle = [0] * group_len
            GDistance = [0] * group_len
            GOC = [0] * group_len
            GPupil = [0] * group_len
            GIris = [0] * group_len
            GSkybox_expo = [0] * group_len
            GSkybox_rot = [0] * group_len
            GLight = [0] * group_len
            GAmbien = [0] * group_len
            GHeadposeX = [0] * group_len
            GHeadposeY = [0] * group_len
            GGroundTruth = [0] * group_len
            GStrangeDistBot = [0] * group_len
            GStrangeDistTop = [0] * group_len
            GStrangeDetect = [0] * group_len
            GDist_x = [0] * group_len
            GMC = [0] * group_len

            for i in range(0, len(unsafe_LR)):
                if(cc[i]==label):
                    group_indices[j] = i
                    j = j + 1

            k = 0
            for index in group_indices:
                Files[k] = unsafe_LR[index]
                GDNN_Result[k] = DNN_Result[index]
                GAngle[k] = Angle[index]
                if(Angle[index] == 0 or Angle[index] == 45 or Angle[index] == 135 or Angle[index] == 90 or Angle[index] == 180 or Angle[index] == 225 or Angle[index] == 270 or Angle[index] == 315 ):
                    GMC[k] = 1
                #if (Angle[index] >= 0 and Angle[index] < 5 or Angle[index] >= 337.5 and Angle[index] <= 340) or (Angle[index] >= 157.5 and Angle[index] < 160.5):
                #    if Dist_x[index] <= 29:
                #        GMC[k] = 1
                if(GroundTruth[index] == "MiddleCenter"):
                    GMC[k] = 1
                GDistance[k] = Distance[index]
                GOC[k] = OC[index]
                GPupil[k] = Pupil[index]
                GIris[k] = Iris[index]
                GSkybox_expo[k] = Skybox_expo[index]
                GSkybox_rot[k] = Skybox_rot[index]
                GLight[k] = Light[index]
                GAmbien[k] = Ambien[index]
                GHeadposeX[k] = HeadposeX[index]
                GHeadposeY[k] = HeadposeY[index]
                GGroundTruth[k] = GroundTruth[index]
                GStrangeDistBot[k] = StrangeDistBot[index]
                GStrangeDistTop[k] = StrangeDistTop[index]
                GStrangeDetect[k] = StrangeDetect[index]
                GDist_x[k] = Dist_x[index]
                k = k + 1

            GstrangeDetectx = sum(GStrangeDetect)/len(GStrangeDetect)
            GMCx = sum(GMC)/len(GMC)
            avg_G = do_avg(GAngle, GDistance, GPupil, GIris, GSkybox_expo, GSkybox_rot, GLight, GAmbien, GHeadposeX, GHeadposeY,
                           GStrangeDistBot, GStrangeDistTop, GDist_x, n_param)
            var_G = do_var(GAngle, GDistance, GPupil, GIris, GSkybox_expo, GSkybox_rot, GLight, GAmbien, GHeadposeX, GHeadposeY,
                           GStrangeDistBot, GStrangeDistTop,  GDist_x, n_param)
            medad_G = do_medad(GAngle, GDistance, GPupil, GIris, GSkybox_expo, GSkybox_rot, GLight, GAmbien, GHeadposeX, GHeadposeY,
                           GStrangeDistBot, GStrangeDistTop,  GDist_x, n_param)
            meanad_G = do_meanad(GAngle, GDistance, GPupil, GIris, GSkybox_expo, GSkybox_rot, GLight, GAmbien, GHeadposeX, GHeadposeY,
                           GStrangeDistBot, GStrangeDistTop, GDist_x,  n_param)
            med_G = do_med(GAngle, GDistance, GPupil, GIris, GSkybox_expo, GSkybox_rot, GLight, GAmbien, GHeadposeX, GHeadposeY,
                           GStrangeDistBot, GStrangeDistTop, GDist_x,  n_param)

            varp_G = do_perc(var_G, var_A)
            medp_G = do_perc(meanad_G, meanad_A)
            madp_G = do_perc(medad_G, medad_A)

            var_G[0], varp_G[0] = degree_var(GAngle, Angle)

            df1['FileName'] = pd.Series(Files)
            df1 = df1.append({"FileName": "Group" + str(label)}, ignore_index=True)
            df1['DNN_Result'] = pd.Series(GDNN_Result)
            df1['GroundTruth'] = pd.Series(GGroundTruth)
            df1['Open/Closed'] = pd.Series(GOC)
            df1['Angle'] = pd.Series(GAngle)
            df1['Distance'] = pd.Series(GDistance)
            df1['Pupil Size'] = pd.Series(GPupil)
            df1['Iris Size'] = pd.Series(GIris)
            df1['Skybox Exposure'] = pd.Series(GSkybox_expo)
            df1['Skybox Rotation'] = pd.Series(GSkybox_rot)
            df1['Light Intenisty'] = pd.Series(GLight)
            df1['Ambient Intenisty'] = pd.Series(GAmbien)
            df1['HeadPoseX'] = pd.Series(GHeadposeX)
            df1['HeadPoseY'] = pd.Series(GHeadposeY)
            df1['StrangeDistBot'] = pd.Series(GStrangeDistBot)
            df1['StrangeDistTop'] = pd.Series(GStrangeDistTop)
            df1['StrangeDetect'] = pd.Series(GStrangeDetect)
            df1['MiddleCenter'] = pd.Series(GMC)
            df1['Dist_x'] = pd.Series(GDist_x)
            clust = [label] * n_param
            feature_var, val_var, ranks_var, feature_mad, val_mad, ranks_mad, feature_med, val_med, ranks_med = rank(varp_G, medp_G, madp_G)

            avgp_G = avg(varp_G, medp_G, madp_G)
            feat_avg, rank_avg, val_avg = rankavg(avgp_G)
            params = ["Angle", "Distance", "Pupil Size", "Iris Size", "Skybox Exposure",
                 "Skybox Rotation", "Light Intenisty", "Ambient Intenisty", "HeadPoseX",
                 "HeadPoseY","StrangeDistBot","StrangeDistTop", "Dist_x"]
            var_c, mad_c, med_c, avg_c = thresholds(varp_G, madp_G, medp_G, avgp_G)
            below50_var = below30(feature_var, val_var)
            below50_mad = below30(feature_mad, val_mad)
            below50_med = below30(feature_med, val_med)
            below50_avg = below30(feat_avg, val_avg)
            below100_var = below100(feature_var, val_var)
            df3['FileName'] = pd.Series()
            df3['Params'] = pd.Series(params)
            df3['Avg'] = pd.Series(avg_G)
            df3['Median'] = pd.Series(med_G)
            df3['Variance'] = pd.Series(var_G)
            df3['Mean-AD'] = pd.Series(meanad_G)
            df3['Median-AD'] = pd.Series(medad_G)
            df3['Var %'] = pd.Series(varp_G)
            df3['Mean-AD %'] = pd.Series(medp_G)
            df3['Median-AD %'] = pd.Series(madp_G)
            df3['Avg %'] = pd.Series(avgp_G)
            df3['Cluster'] = pd.Series(clust)
            df3['Params_Var'] = pd.Series(feature_var)
            df3['Ranks_Var'] = pd.Series(ranks_var)
            df3['Params_Below50_BasedOnVar'] = pd.Series(below50_var)
            df3['Params_Mad'] = pd.Series(feature_mad)
            df3['Ranks_Mad'] = pd.Series(ranks_mad)
            df3['Params_Below50_BasedOnMad'] = pd.Series(below50_mad)
            df3['Params_Med'] = pd.Series(feature_med)
            df3['Ranks_Med'] = pd.Series(ranks_med)
            df3['Params_Below50_BasedOnMed'] = pd.Series(below50_med)
            df3['Params_Avg'] = pd.Series(feat_avg)
            df3['Ranks_Avg'] = pd.Series(rank_avg)
            df3['Params_Below50_BasedOnAvg'] = pd.Series(below50_avg)
            Tests = ["Border 337.5", "Border 22.5", "Border 67.5", "Border 112.5", "Border 157.5", "Border 202.5", "Border 247.5", "Border 292.5", "Border 220 Horizontal", "Border 160 Horizontal", "Border 20 Vertical", "Border 340 Vertical", "StrangeDist -15", "Distance 25"]
            CheckedTests = [0] * len(Tests)
            tuplex, bool1, bool2, clustpass, CheckedTests = test_tuples(below50_var, CheckedTests, avg_G)
            #tuplex, bool1, bool2, clustpass, CheckedTests = test_tuples(below100_var, CheckedTests, avg_G)
            if(bool1 == True):
                num_below = num_below + 1
            if(bool2 == True):
                num_test = num_test + 1
            num_pass = list(CheckedTests).count(1)
            x = 0
            for (col, lst) in enumerate(tuplex):
                df3[col] = pd.Series(lst)
                if(x == 0):
                    numpass1 = numpass1 + lst[1]
                if(x == 1):
                    numpass2 = numpass2 + lst[1]
                if(x == 2):
                    numpass3 = numpass3 + lst[1]
                if(x == 3):
                    numpass4 = numpass4 + lst[1]
                if(x == 4):
                    numpass5 = numpass5 + lst[1]
                x = x + 1
            #df3['Tests'] = pd.Series(Tests)
            #df3['Success/Fail'] = pd.Series(CheckedTests)
            df3 = df3.append({"FileName": "Group"+str(label)},ignore_index=True)
            df3 = df3.append({"FileName": "%Strange" + str(GstrangeDetectx)}, ignore_index=True)
            df3 = df3.append({"FileName": "%Strange" + str(GMCx)}, ignore_index=True)
            Thresholds = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
            var_c_clusters[label-1] = var_c
            mad_c_clusters[label-1] = mad_c
            med_c_clusters[label-1] = med_c
            avg_c_clusters[label-1] = avg_c

            avganalysis1[label-1] = doAvgDisp(varp_G, len(params))
            avganalysis2[label-1] = doAvgDisp(varp_G, 1)
            avganalysis3[label-1] = doAvgDisp(varp_G, 3)
            avganalysis4[label-1] = doAvgDisp(varp_G, 4)
            avganalysis5[label-1] = doAvgDispBelow50(varp_G)
            #avganalysis2[label-1] = sum(madp_G)/len(params)
            #avganalysis3[label-1] = sum(medp_G)/len(params)
            #avganalysis4[label-1] = sum(avgp_G)/len(params)
            df4 = pd.concat([df4, df3])
            df2 = pd.concat([df2, df1])
            sucess1 = sucess1 + sum(CheckedTests)
            reduced_param_val[label-1] = min(varp_G)
        else:
            reduced_param_val[label - 1] = 0
            discarded = discarded + 1
    df6 = pd.DataFrame()
    df5 = pd.DataFrame()
    arr = [0] * 17
    arr[0] = n_clusters
    if((n_clusters - discarded) > 1):
        sucessx = sucess1/n_clusters
        sucess2 = sucess1/(3*n_clusters)
        sucess3 = clustpass/n_clusters
        avganalysist1 = sum(avganalysis1)/len(avganalysis1)
        avganalysist2 = sum(avganalysis2)/len(avganalysis2)
        avganalysist3 = sum(avganalysis3)/len(avganalysis3)
        avganalysist4 = sum(avganalysis4)/len(avganalysis4)
        avganalysist5 = 0
        n = 0
        for i in avganalysis5:
            if i > 0:
                avganalysist5 += i
                n += 1
        avganalysist5 /= n
        arr = [n_clusters, avganalysist1, avganalysist2, avganalysist3, avganalysist4, avganalysist5, numpass1, numpass2, numpass3, numpass4, numpass5, sucessx, sucess2, num_below, num_test, num_below/n_clusters, num_test/n_clusters]
        print(arr)
        #rank5 = rank_clust(n_clusters, avg_c_clusters, 0)
        #rank4 = rank_clust(n_clusters, avg_c_clusters, 1)
        #rank3 = rank_clust(n_clusters, avg_c_clusters, 2)
        #rank2 = rank_clust(n_clusters, avg_c_clusters, 3)
        #rank1 = rank_clust(n_clusters, avg_c_clusters, 4)
        #df4['Rank5'] = pd.Series(rank5)
        #df4['Rank4'] = pd.Series(rank4)
        #df4['Rank3'] = pd.Series(rank3)
        #df4['Rank2'] = pd.Series(rank2)
        #df4['Rank1'] = pd.Series(rank1)

        #df5['Thresholds_Varp'] = Thresholds
        for label in range(1, n_clusters+1):
            df5['Varp_Cluster' + str(label)] = var_c_clusters[label-1]
        #df5['Thresholds_Madp'] = Thresholds
        for label in range(1, n_clusters+1):
            df5['Madp_Cluster' + str(label)] = mad_c_clusters[label-1]
        #df5['Thresholds_Medp'] = Thresholds
        for label in range(1, n_clusters+1):
            df5['Medp_Cluster' + str(label)] = med_c_clusters[label-1]
        #df5['Thresholds_Avgp'] = Thresholds
        for label in range(1, n_clusters+1):
            df5['Avgp_Cluster' + str(label)] = avg_c_clusters[label-1]
        #df5.drop(df5.index[1])
        #df5 = pd.concat([df5a, df5b, df5c, df5d])
    #df4.drop(df4.index[1])
    ind = sum(reduced_param_val)/(n_clusters-discarded)
    return df2, df4, df5, arr, df6, discarded, ind


def doAvgDisp(varp_G, n):
    varp_G.sort(reverse=False)
    var = 0
    for i in range(0, n):
        var += varp_G[i]
    return (1 - (var/n)) * 100.0


def doAvgDispBelow50(varp_G):
    varp_G.sort(reverse=False)
    var = 0
    n = 0
    for i in range(0, len(varp_G)):
        if varp_G[i] < 0.5:
            var += varp_G[i]
            n += 1
    if n == 0:
        return 0
    return (1 - (var/n)) * 100.0


def rank_clust(n_clusters, avg_c_clusters, r):
    rank = [0] * n_clusters
    k = 0
    for label in range(0, n_clusters):
        if(avg_c_clusters[label][4] == r):
            rank[k] = label
            k = k+1
    return rank


def test_tuples(below50_avg, CheckedTests, avg_G):
    bool1 = False
    bool2 = False
    val1 = 0
    val2 = 0
    val3 = 0
    val4 = 0
    val5 = 0
    margin1 = 0
    margin2 = 0
    margin3 = 0
    margin4 = 0
    margin5 = 0
    dist1 = 0
    dist2 = 0
    dist3 = 0
    dist4 = 0
    dist5 = 0
    boundary1 = 0
    boundary2 = 0
    boundary3 = 0
    boundary4 = 0
    boundary5 = 0
    pass1 = 0
    pass2 = 0
    pass3 = 0
    pass4 = 0
    pass5 = 0
    param1 = "Angle"
    param2 = "H_Headpose"
    param3 = "V_Headpose"
    param4 = "StrangeDist"
    param5 = "Distance"
    clustpass = 0
    if (below50_avg[0] != 0):
        bool1 = True
    for param in below50_avg:
        if (param == "Angle"):
            margin1 = 45*0.25
            if (((337.5-margin1)) < avg_G[0] < ((337.5+margin1))):
                CheckedTests[0] = 1
                bool2 = True
                boundary1 = 337.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
            elif (((22.5-margin1))< avg_G[0] < ((22.5+margin1))):
                CheckedTests[1] = 1
                bool2 = True
                boundary1 = 22.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
            elif (((67.5-margin1)) < avg_G[0] < ((67.5+margin1))):
                CheckedTests[2] = 1
                bool2 = True
                boundary1 = 67.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
            elif ((112.5-margin1) < avg_G[0] < ((112.5+margin1))):
                CheckedTests[3] = 1
                bool2 = True
                boundary1 = 112.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
            elif (((157.5-margin1)) < avg_G[0] < ((157.5+margin1))):
                CheckedTests[4] = 1
                bool2 = True
                boundary1 = 157.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
            elif (((202.5-margin1)) < avg_G[0] < ((202.5+margin1))):
                CheckedTests[5] = 1
                bool2 = True
                boundary1 = 202.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
            elif (((247.5-margin1)) < avg_G[0] < ((247.5+margin1))):
                CheckedTests[6] = 1
                bool2 = True
                boundary1 = 247.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
            elif (((292.5-margin1)) < avg_G[0] < (292.5+margin1)):
                CheckedTests[7] = 1
                bool2 = True
                boundary1 = 292.5
                pass1 = pass1 + 1
                val1 = avg_G[0]
                margin1 = val1 / boundary1
                clustpass = clustpass + 1
                dist1 = val1 - boundary1
        margin2 = (60*0.25)
        if (param == "H_Headpose"):
            if ((220-margin2)) < avg_G[9] < ((220)):
                CheckedTests[8] = 1
                bool2 = True
                boundary2 = 220
                pass2 = pass2 + 1
                val2 = avg_G[9]
                margin2 = val2 / boundary2
                clustpass = clustpass + 1
                dist2 = val2 - boundary2
            elif (((160)) < avg_G[9] < ((160+margin2))):
                CheckedTests[9] = 1
                bool2 = True
                boundary2 = 160
                pass2 = pass2 + 1
                val2 = avg_G[9]
                margin2 = val2 / boundary2
                clustpass = clustpass + 1
                dist2 = val2 - boundary2
        margin3 = 40*0.25
        if (param == "V_Headpose"):
            if (((20-margin3)) < avg_G[8] < ((20))):
                CheckedTests[10] = 1
                bool2 = True
                boundary3 = 20
                pass3 = pass3 + 1
                val3 = avg_G[8]
                clustpass = clustpass + 1
                margin3 = val3 / boundary3
                dist3 = val3 - boundary3
            elif (((340)) < avg_G[8] < ((340+margin3))):
                CheckedTests[11] = 1
                bool2 = True
                clustpass = clustpass + 1
                boundary3 = 340
                pass3 = pass3 + 1
                val3 = avg_G[8]
                margin3 = val3 / boundary3
                dist3 = val3 - boundary3
        if ((param == "StrangeDistBot")):
            if (avg_G[10] < -16):
                CheckedTests[12] = 1
                bool2 = True
                boundary4 = -14
                pass4 = pass4 + 1
                val4 = avg_G[10]
                clustpass = clustpass + 1
                margin4 = val4 / boundary4
                dist4 = val4 - boundary4
        if (param == "StrangeDistTop"):
            if (avg_G[11] <  -16):
                CheckedTests[12] = 1
                bool2 = True
                boundary4 = -14
                pass4 = pass4 + 1
                val4 = avg_G[11]
                clustpass = clustpass + 1
                margin4 = val4 / boundary4
                dist4 = val4 - boundary4
        margin5 = 64*0.25
        if (param == "Distance"):
            if ((20-margin5) < avg_G[1] < (20+margin5)):
                CheckedTests[13] = 1
                bool2 = True
                boundary5 = 25
                pass5 = pass5 + 1
                val5 = avg_G[1]
                clustpass = clustpass + 1
                margin5 = val5 / boundary5
                dist5 = val5 - boundary5
    tuple1 = (param1, pass1, val1, boundary1, dist1, margin1)
    tuple2 = (param2, pass2, val2, boundary2, dist2, margin2)
    tuple3 = (param3, pass3, val3, boundary3, dist3, margin3)
    tuple4 = (param4, pass4, val4, boundary4, dist4, margin4)
    tuple5 = (param5, pass5, val5, boundary5, dist5, margin5)
    tuplex = (tuple1, tuple2, tuple3, tuple4, tuple5)
    return tuplex, bool1, bool2, clustpass, CheckedTests


def below30(feat, values):
    k = 0
    below30 = [0] * len(feat)
    for i in range(0, len(feat)):
        if (values[i] < 0.51):
            below30[k] = feat[i]
            k = k + 1
    return below30


def below100(feat, values):
    k = 0
    below30 = [0] * len(feat)
    for i in range(0, len(feat)):
        if (values[i] < 1):
            below30[k] = feat[i]
            k = k + 1
    return below30


def avg(varp, madp, medp):
    avgp = [0] * len(varp)
    for i in range(0, len(varp)):
        avgp[i] = (varp[i] + madp[i] + medp[i]) / 3
    return avgp


def degree_var(Group, All):
    GroupM = [0] * len(Group)
    AllM = [0] * len(All)
    for i in range(0, len(Group)):
        GroupM[i] = (Group[i] * (math.pi / 180))
    for i in range(0, len(All)):
        AllM[i] = (All[i] * (math.pi / 180))
    gvar = sc.circvar(GroupM, high=2 * math.pi, low=0)
    avar = sc.circvar(AllM, high=2 * math.pi, low=0)
    pvar = gvar / avar
    return gvar, pvar


def thresholds(varp_G, madp_G, medp_G, avgp_G):
    var_c = [0] * 10
    mad_c = [0] * 10
    med_c = [0] * 10
    avg_c = [0] * 10
    for k in range(1, 11):
        p = 0
        p2 = 0
        p3 = 0
        p4 = 0
        for i in range(0, len(varp_G)):
            if (varp_G[i] < k / 10):
                p += 1
            if (madp_G[i] < k / 10):
                p2 += 1
            if (medp_G[i] < k / 10):
                p3 += 1
            if (avgp_G[i] < k / 10):
                p4 += 1
        var_c[k - 1] = p
        mad_c[k - 1] = p2
        med_c[k - 1] = p3
        avg_c[k - 1] = p4
    return var_c, mad_c, med_c, avg_c


def rankavg(avgp):
    avgp_lst = list(enumerate(avgp))
    avgp_lst.sort(key=lambda x: x[1])
    feature_var = [0] * len(avgp)
    ranks_var = [0] * len(avgp)
    values_var = [0] * len(avgp)
    rank = 1
    for index, val in avgp_lst:
        feature_var[rank-1] = getfeature(index)
        ranks_var[rank-1] = rank
        values_var[rank-1] = val
        rank = rank + 1
    return feature_var, ranks_var, values_var


def rank(varp, medp, madp):
    varp_lst = list(enumerate(varp))
    medp_lst = list(enumerate(medp))
    madp_lst = list(enumerate(madp))
    varp_lst.sort(key=lambda x: x[1])
    medp_lst.sort(key=lambda x: x[1])
    madp_lst.sort(key=lambda x: x[1])
    feature_var = [0] * len(varp)
    ranks_var = [0] * len(varp)
    values_var = [0] * len(varp)
    feature_med = [0] * len(varp)
    ranks_med = [0] * len(varp)
    values_med = [0] * len(varp)
    feature_mad = [0] * len(varp)
    ranks_mad = [0] * len(varp)
    values_mad = [0] * len(varp)
    rank = 1
    for index, val in varp_lst:
        feature_var[rank-1] = getfeature(index)
        ranks_var[rank-1] = rank
        values_var[rank-1] = val
        rank = rank + 1
    rank = 1
    for index, val in medp_lst:
        feature_med[rank-1] = getfeature(index)
        ranks_med[rank-1] = rank
        values_med[rank-1] = val
        rank = rank + 1
    rank = 1
    for index, val in madp_lst:
        feature_mad[rank-1] = getfeature(index)
        ranks_mad[rank-1] = rank
        values_mad[rank-1] = val
        rank = rank + 1
    return feature_var, values_var, ranks_var, feature_mad, values_mad, ranks_mad, feature_med, values_med, ranks_med


def lowest(listX):
    varp_lst = list(enumerate(listX))
    varp_lst.sort(key=lambda x: x[1])
    varp_feat = getfeature(varp_lst[0][0], '')
    varp_featv = varp_lst[0][1]
    varp_feat2 = getfeature(varp_lst[1][0], '')
    varp_feat2v = varp_lst[1][1]
    return [varp_feat, varp_featv, varp_feat2, varp_feat2v]


def lowestg(listX, listY):
    varp_lst = list(enumerate(listX))

    varp_lst.sort(key=lambda x: x[1])
    varp_feat = getfeature(varp_lst[0][0], '')
    varp_featv = listY[varp_lst[0][0]]
    varp_feat2 = getfeature(varp_lst[1][0], '')
    varp_feat2v = listY[varp_lst[1][0]]
    return [varp_feat, varp_featv, varp_feat2, varp_feat2v]


def getfeature(i):
    if(i==0):
        feat='Angle'
    elif(i==1):
        feat = 'Distance'
    elif(i==2):
        feat = 'Pupil Size'
    elif(i==3):
        feat = 'Iris Size'
    elif(i==4):
        feat = 'Skybox Exposure'
    elif(i==5):
        feat = 'Skybox Rotation'
    elif(i==6):
        feat = 'Light Intenisty'
    elif(i==7):
        feat = 'Ambien Intenisty'
    elif(i==8):
        feat = 'V_Headpose'
    elif(i==9):
        feat = 'H_Headpose'
    elif(i==10):
        feat = 'StrangeDistBot'
    elif(i==11):
        feat = 'StrangeDistTop'
    elif(i==12):
        feat = 'Dist_x'

    return feat


def do_min(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, paramlen):
    min_A = [0] * paramlen
    min_A[0] = min(Angle)
    min_A[1] = min(Distance)
    min_A[2] = min(Pupil)
    min_A[3] = min(Iris)
    min_A[4] = min(Skybox_expo)
    min_A[5] = min(Skybox_rot)
    min_A[6] = min(Light)
    min_A[7] = min(Ambien)
    min_A[8] = min(HeadposeX)
    min_A[9] = min(HeadposeY)
    min_A[10] = min(StrangeDistBot)
    min_A[11] = min(StrangeDistTop)
    min_A[12] = min(Dist_x)
    return min_A


def do_max(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, paramlen):
    max_A = [0] * paramlen
    max_A[0] = max(Angle)
    max_A[1] = max(Distance)
    max_A[2] = max(Pupil)
    max_A[3] = max(Iris)
    max_A[4] = max(Skybox_expo)
    max_A[5] = max(Skybox_rot)
    max_A[6] = max(Light)
    max_A[7] = max(Ambien)
    max_A[8] = max(HeadposeX)
    max_A[9] = max(HeadposeY)
    max_A[10] = max(StrangeDistBot)
    max_A[11] = max(StrangeDistTop)
    max_A[12] = max(Dist_x)
    return max_A


def do_avg(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, paramlen):
    avg_A = [0] * paramlen
    avg_A[0] = sum(Angle) / len(Angle)
    avg_A[1] = sum(Distance) / len(Distance)
    avg_A[2] = sum(Pupil) / len(Pupil)
    avg_A[3] = sum(Iris) / len(Iris)
    avg_A[4] = sum(Skybox_expo) / len(Skybox_expo)
    avg_A[5] = sum(Skybox_rot) / len(Skybox_rot)
    avg_A[6] = sum(Light) / len(Light)
    avg_A[7] = sum(Ambien) / len(Ambien)
    avg_A[8] = sum(HeadposeX) / len(HeadposeX)
    avg_A[9] = sum(HeadposeY) / len(HeadposeY)
    avg_A[10] = sum(StrangeDistBot) / len(StrangeDistBot)
    avg_A[11] = sum(StrangeDistTop) / len(StrangeDistTop)
    avg_A[12] = sum(Dist_x) / len(Dist_x)
    return avg_A


def do_med(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, paramlen):
    med_A = [0] * paramlen
    med_A[0] = stat.median(Angle)
    med_A[1] = stat.median(Distance)
    med_A[2] = stat.median(Pupil)
    med_A[3] = stat.median(Iris)
    med_A[4] = stat.median(Skybox_expo)
    med_A[5] = stat.median(Skybox_rot)
    med_A[6] = stat.median(Light)
    med_A[7] = stat.median(Ambien)
    med_A[8] = stat.median(HeadposeX)
    med_A[9] = stat.median(HeadposeY)
    med_A[10] = stat.median(StrangeDistBot)
    med_A[11] = stat.median(StrangeDistTop)
    med_A[12] = stat.median(Dist_x)
    return med_A


def do_var(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, paramlen):
    var_A = [0] * paramlen
    var_A[0] = stat.variance(Angle)
    var_A[1] = stat.variance(Distance)
    var_A[2] = stat.variance(Pupil)
    var_A[3] = stat.variance(Iris)
    var_A[4] = stat.variance(Skybox_expo)
    var_A[5] = stat.variance(Skybox_rot)
    var_A[6] = stat.variance(Light)
    var_A[7] = stat.variance(Ambien)
    var_A[8] = stat.variance(HeadposeX)
    var_A[9] = stat.variance(HeadposeY)
    var_A[10] = stat.variance(StrangeDistBot)
    var_A[11] = stat.variance(StrangeDistTop)
    var_A[12] = stat.variance(Dist_x)
    return var_A


def do_medad(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, paramlen):
    medad_A = [0] * paramlen
    medad_A[0] = sc.median_absolute_deviation(Angle)
    medad_A[1] = sc.median_absolute_deviation(Distance)
    medad_A[2] = sc.median_absolute_deviation(Pupil)
    medad_A[3] = sc.median_absolute_deviation(Iris)
    medad_A[4] = sc.median_absolute_deviation(Skybox_expo)
    medad_A[5] = sc.median_absolute_deviation(Skybox_rot)
    medad_A[6] = sc.median_absolute_deviation(Light)
    medad_A[7] = sc.median_absolute_deviation(Ambien)
    medad_A[8] = sc.median_absolute_deviation(HeadposeX)
    medad_A[9] = sc.median_absolute_deviation(HeadposeY)
    medad_A[10] = sc.median_absolute_deviation(StrangeDistBot)
    medad_A[11] = sc.median_absolute_deviation(StrangeDistTop)
    medad_A[12] = sc.median_absolute_deviation(Dist_x)
    return medad_A


def do_meanad(Angle, Distance, Pupil, Iris, Skybox_expo, Skybox_rot, Light, Ambien, HeadposeX, HeadposeY, StrangeDistBot, StrangeDistTop, Dist_x, paramlen):
    meanad_A = [0] * paramlen
    series1 = pd.Series(Angle)
    meanad_A[0] = series1.mad()
    series2 = pd.Series(Distance)
    meanad_A[1] = series2.mad()
    series3 = pd.Series(Pupil)
    meanad_A[2] = series3.mad()
    series4 = pd.Series(Iris)
    meanad_A[3] = series4.mad()
    series5 = pd.Series(Skybox_expo)
    meanad_A[4] = series5.mad()
    series6 = pd.Series(Skybox_rot)
    meanad_A[5] = series6.mad()
    series7 = pd.Series(Light)
    meanad_A[6] = series7.mad()
    series8 = pd.Series(Ambien)
    meanad_A[7] = series8.mad()
    series9 = pd.Series(HeadposeX)
    meanad_A[8] = series9.mad()
    series10 = pd.Series(HeadposeY)
    meanad_A[9] = series10.mad()
    series11 = pd.Series(StrangeDistBot)
    meanad_A[10] = series11.mad()
    series12 = pd.Series(StrangeDistTop)
    meanad_A[11] = series12.mad()
    series13 = pd.Series(Dist_x)
    meanad_A[12] = series13.mad()
    return meanad_A


def do_perc(var_G, var_A):
    varp_G = [0] * len(var_G)
    for i in range(0, len(var_G)):
        varp_G[i] = var_G[i]/var_A[i]
    return varp_G


def getParams(testCSV, datasetNpyPath, outDir, clsWithAssImages, outFile):
    dataset = np.load(datasetNpyPath, allow_pickle=True)
    dataset = dataset.item()
    x_config = dataset["config"]
    allDictParams = {}
    outFile = open(outFile, 'w')
    strMerge = "image,clusterID"
    for param in x_config[0]:
        if isinstance(x_config[0][param], str):
            continue
        elif isinstance(x_config[0][param], float):
            allDictParams[param] = []
            strMerge += "," + str(param)
        else:
            if len(x_config[0][param]) > 1:
                i = 0
                for component in x_config[0][param]:
                    allDictParams[param + "_" + str(i)] = list()
                    strMerge += "," + str(param + "_" + str(i))
                    i += 1
    #imagesList = {}
    redDict = {}
    clusterDictParams = {}
    allImagesList = list()
    strMerge = strMerge + "\r\n"
    outFile.writelines(strMerge)
    imagesList = []

    imageList = pd.read_csv(testCSV)
    #clsWithAssImages = {'clusters': {'All':{'members':[]}}}
    for index, row in imageList.iterrows():
        #clsWithAssImages['clusters']['All']['members'].append(basename(row["image"]))

    #for clusterID in clsWithAssImages['clusters']:  # RCC

        clusterID = 0
        #imagesList[clusterID] = []
        redDict[clusterID] = {}
        #dirPathTemp = join(outDir, str(clusterID))
        clusterDictParams[clusterID] = {}

        for param in x_config[0]:
            if isinstance(x_config[0][param], str):
                continue
            elif isinstance(x_config[0][param], float):
                clusterDictParams[clusterID][param] = []
            else:
                if len(x_config[0][param]) > 1:
                    i = 0
                    for component in x_config[0][param]:
                        clusterDictParams[clusterID][param + "_" + str(i)] = []
                        i += 1
        classe = 0
        for clusterID in clsWithAssImages['clusters']:
            if ("Test_" + basename(row["image"]).split(".")[0] + "_" + basename(dirname(row["image"]))) in clsWithAssImages['clusters'][clusterID]['members']:
                if row["result"] == "Wrong":
                #classe = clusterID
                    classe = 1
        clusterID = 0
        member = basename(row["image"])
        #imageName = member.split("_")[1] + ".png"
        imageName = member
        imagesList.append(imageName)
        strMerge = imageName + "," + str(classe)
        #imageParams = x_config[int(imageName.split(".")[0]) - 16013]
        #imageParams = x_config[int(imageName.split(".")[0])]
        #imageParams = x_config[int(imageName.split(".")[0]) - 1]
        imageParams = x_config[int(imageName.split(".")[0]) - 1]
        for param in imageParams:
            if isinstance(imageParams[param], str):
                continue
            elif isinstance(imageParams[param], float):
                allDictParams[param].append(float(imageParams[param]))
                clusterDictParams[clusterID][param].append(float(imageParams[param]))
            else:
                if len(imageParams[param]) > 1:

                    #if param == "head_pose":
                        #if imageParams[param] <
                    i = 0
                    for component in imageParams[param]:
                        allDictParams[param + "_" + str(i)].append(float(component))
                        clusterDictParams[clusterID][param + "_" + str(i)].append(float(component))
                        i += 1
        for param in clusterDictParams[clusterID]:
            strMerge += "," + str(clusterDictParams[clusterID][param][len(clusterDictParams[clusterID][param])-1])
        strMerge = strMerge + "\r\n"
        outFile.writelines(strMerge)
    for param in allDictParams:
        print(param)
        print(min(allDictParams[param]))
        print(max(allDictParams[param]))

    outFile.close()
    return
            # print(clusterID, folderName, len(imagesList[clusterID][folderName]))
    # conceptsThresholds = pd.DataFrame()
    # lst = list()
    #     lst.append(float(i/10))
    #    for clusterID in clsWithAssImages['clusters']:
    #        conceptsThresholds[clusterID] = list()
    # conceptsThresholds["Threshold"] = pd.Series(lst)
    for i in range(1, 11):
        Threshold = i / 10
        print(str(Threshold*100) + "%")
        for clusterID in clsWithAssImages['clusters']:  # RCC
            n = 0
            numReducedParams = 0
            redFlag = False
            redList = list()
            conceptsParams = pd.DataFrame()
            dirPathTemp = join(outDir, str(clusterID))
            diffList = list()
            for param in clusterDictParams[clusterID]:
                VarRedC = 1.0
                if len(clusterDictParams[clusterID][param]) > 1:
                    if isinstance(clusterDictParams[clusterID][param][0], str):
                        VarRedC = 1.0
                        # param is string, variance must be manually implemented
                        # FIXME
                    else:
                        # print(stat.variance(allDictParams[param]))
                        if stat.variance(allDictParams[param]) == 0.0:
                            VarRedC = 1.0
                            #print(param, "is constant")
                        else:
                            VarRedC = \
                                stat.variance(clusterDictParams[clusterID][param]) / stat.variance(allDictParams[param])
                if VarRedC < Threshold:
                    if not isinstance(clusterDictParams[clusterID][param][0], str):
                        redFlag = True
                        redList.append(param)
                        # print(clusterID, folderName, Threshold, param)
                        diffList.append(param)
                        diffList.append(VarRedC)
                        diffList.append(min(clusterDictParams[clusterID][param]))
                        diffList.append(max(clusterDictParams[clusterID][param]))
                        diffList.append(np.percentile(clusterDictParams[clusterID][param], 25))  # FIXME
                        diffList.append(np.percentile(clusterDictParams[clusterID][param], 75))  # FIXME
                if redFlag:
                    numReducedParams += 1
                    redFlag = False
                n += 1
                # conceptsParams = pd.Series(diffList)
            print(str(clusterID))
            print(redList)
            # conceptsThresholds[clusterID].append(math.ceil(100*(numReducedConcepts/n)))
            # writer = pd.ExcelWriter(join(dirPathTemp, "conceptsParams_" + str(Threshold) + ".xlsx"),
            #                        engine='xlsxwriter')
            # writer.book.use_zip64()
            # conceptsParams.to_excel(writer)
            # writer.close()


    return