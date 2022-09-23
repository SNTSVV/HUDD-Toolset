from imports import os, cv2, json, np, math
def label():
    outputPath = ""
    for file in os.listdir(outputPath):
        if file.endswith(".jpg"):
            fileClass = str(file.split("_")[0])
            fileName = str(file.split("_")[1])
            img = cv2.imread(orig_dir + "/" + fileClass + "/" + fileName + ".jpg")
            json_fn = jsonx + "/" + fileName + ".json"
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
            angle, point_A, point_B, point_C = computeAngle(data, img)
            # angle, milieu_x, milieu_y, intersection, dist_x, dist_y = labelimages.executePiplineForInformation(json_fn)
            if (mode == 'GD'):
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

            if (mode == 'OC'):
                if (dist < 20):
                    classe = 'C'
                else:
                    classe = 'O'

def process_json_list(json_list,img):
    ldmks = [eval(s) for s in json_list]
    return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

def getMiddelX(data, img):
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    return milieu(int(ldmks_interior_margin[0][0]), int(ldmks_interior_margin[0][1]),
                      int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0]),
                      int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1]))
def computeAngle(data,img):
    ldmks_iris = process_json_list(data['iris_2d'],img)
    look_vec = list(eval(data['eye_details']['look_vec']))
    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    #print(look_vec)
    look_vec[1] = -look_vec[1]
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))
    angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
                                                                                      point_B[1] - point_A[1]);

    angle = (angle * 180) / math.pi

    while (angle < 0):
        angle = angle + 360

    return angle, point_A, point_B, point_C


def milieu(x1, y1, x2, y2):
 x = (x1 + x2) / 2
 y = (y1 + y2) / 2
 return [x, y]


# coding=utf8
from imports import sys, ntpath, configparser, os, shutil, math, glob, json, np, cv2, isdir, isfile

debug = True

CLASSIFY_WITH_YAW_PICTH = 1
CLASSIFY_WITH_ANGLE = 0

CLASSIFICATION_METHOD = CLASSIFY_WITH_YAW_PICTH


def readStringVar(config, varCat, varName):
    raw = config.get(varCat, varName)
    if (raw.strip().lower() == "none"):
        return None
    else:
        return raw


def readBoolVar(config, varCat, varName):
    raw = config.get(varCat, varName)
    if (raw.strip().lower() == "none"):
        return None
    else:
        if (raw.strip().lower() == "true"):
            return True
        else:
            return False


def milieu(x1, y1, x2, y2):
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return [x, y]


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]


def process_json_list(json_list, img):
    ldmks = [eval(s) for s in json_list]
    return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])


def getIntersection(json_fn):
    img = cv2.imread("%s.jpg" % json_fn[:-5])
    data_file = open(json_fn)
    data = json.load(data_file)

    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)

    return findIntersection(int(ldmks_interior_margin[0][0]), int(ldmks_interior_margin[0][1]),
                            int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0]),
                            int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1]),
                            int(ldmks_interior_margin[4][0]), int(ldmks_interior_margin[4][1]),
                            int(ldmks_interior_margin[12][0]), int(ldmks_interior_margin[12][1])), data, img


def getMiddelX(data, img):
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    return milieu(int(ldmks_interior_margin[0][0]), int(ldmks_interior_margin[0][1]),
                  int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0]),
                  int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1]))


def getMiddelY(data, img):
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    return milieu(int(ldmks_interior_margin[4][0]), int(ldmks_interior_margin[4][1]),
                  int(ldmks_interior_margin[12][0]), int(ldmks_interior_margin[12][1]))


def testdraw(data, img, path, fileName):
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    for ldmk in np.vstack([ldmks_interior_margin[0], ldmks_interior_margin[4], ldmks_interior_margin[12],
                           ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    x1 = int(ldmks_interior_margin[0][0])
    y1 = int(ldmks_interior_margin[0][1])
    x2 = int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0])
    y2 = int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1])
    x3 = int(ldmks_interior_margin[4][0])
    y3 = int(ldmks_interior_margin[4][1])
    x4 = int(ldmks_interior_margin[12][0])
    y4 = int(ldmks_interior_margin[12][1])
    intersection = findIntersection(x1, y1, x2, y2, x3, y3, x4, y4)

    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    ldmks_caruncle = process_json_list(data['caruncle_2d'], img)
    ldmks_iris = process_json_list(data['iris_2d'], img)

    # Draw black background points and lines
    # for ldmk in np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]]):
    #    cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (0, 0, 0), -1)
    # cv2.polylines(img, np.array([ldmks_interior_margin[:, :2]], int), True, (0, 0, 0), 2)
    # cv2.polylines(img, np.array([ldmks_iris[:, :2]], int), True, (0, 0, 0), 2)

    # Draw green foreground points and lines
    for ldmk in np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    cv2.polylines(img, np.array([ldmks_interior_margin[:, :2]], int), True, (0, 255, 0), 1)
    cv2.polylines(img, np.array([ldmks_iris[:, :2]], int), True, (0, 255, 0), 1)

    look_vec = list(eval(data['eye_details']['look_vec']))

    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    look_vec[1] = -look_vec[1]
    cv2.line(img, tuple(eye_c), tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int)), (0, 0, 0), 3)
    cv2.line(img, tuple(eye_c), tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int)), (0, 255, 255), 3)
    # cv2.imshow("syntheseyes_img", img)
    # cv2.imwrite("annotated_%s.png"%json_fn[:-5], img)
    # cv2.waitKey(1)
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))
    # cv2.line(img, point_A, point_B, (0, 0, 0), 3)
    # cv2.line(img, point_A, point_B, (0, 255, 255), 2)
    angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
                                                                                      point_B[1] - point_A[1])
    angle = (angle * 180) / math.pi
    while (angle < 0):
        angle = angle + 360
    # print(angle)
    cv2.imwrite(path + "/" + fileName + "_Gaze.jpg", img)

    # drawCircles(img, intersection, milieu(x1, y1, x2, y2), milieu(x3, y3, x4, y4))
    # cv2.imshow('ImageWindow', img)
    # cv2.waitKey(10000)


def drawLinesAndPoints(data, img):
    # Draw green foreground points and lines
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    for ldmk in np.vstack([ldmks_interior_margin[0], ldmks_interior_margin[4], ldmks_interior_margin[12],
                           ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    cv2.imshow('ImageWindow', img)
    cv2.waitKey(10000)


def drawCircles(img, intersection, milieu_x, milieu_y):
    cv2.circle(img, (int(intersection[0]), int(intersection[1])), 2, (0, 255, 0), -1)
    cv2.circle(img, (int(milieu_x[0]), int(milieu_x[1])), 2, (0, 255, 0), -1)
    cv2.circle(img, (int(milieu_y[0]), int(milieu_y[1])), 2, (0, 255, 0), -1)


def classifyAngle(data, img):
    ldmks_iris = process_json_list(data['iris_2d'], img)
    look_vec = list(eval(data['eye_details']['look_vec']))
    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    # print(look_vec)
    look_vec[1] = -look_vec[1]
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))

    x_angle = math.atan2(look_vec[0], look_vec[2]);
    y_angle = math.atan2(look_vec[1], look_vec[2]);

    # print(x_angle)
    # print(y_angle)

    # FIXME: the class label should be generated with a big if block that checks the values for x_angle and y_angle

    return "TopLeft"


def trial():
    x = input("Enter x")
    y = input("Enter y")
    z = input("Enter z")

    z = -float(z)

    x_angle = math.atan2(float(x), float(z));
    y_angle = math.atan2(float(y), float(z));

    # print(str(x_angle))
    # print(str(y_angle))


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


def executePipelineToClassifyAngle(json_fn):
    intersection, data, img = getIntersection(json_fn)
    return classifyAngle(data, img)


def executePiplineForInformation(json_fn):
    # print("\n\n\n\n\n\n************************************")
    # print("********   computeANgle IMG "+json_fn)
    intersection, data, img = getIntersection(json_fn)
    milieu_x = getMiddelX(data, img)
    milieu_y = getMiddelY(data, img)
    drawLinesAndPoints(data, img)
    angle, point_A, point_B, point_C = computeAngle(data, img)
    drawCircles(img, intersection, milieu_x, milieu_y)
    dist_x = getDistBetweenTwoPoints(point_A, milieu_x)
    dist_y = getDistBetweenTwoPoints(point_A, milieu_y)
    if (debug):
        name = json_fn + ".angle.jpg"
        cv2.imwrite(name, img)

    return angle, milieu_x, milieu_y, intersection, dist_x, dist_y


def getLabelFromSinglePhotoYawPitch(json_fn):
    return executePipelineToClassifyAngle(json_fn)


def getLabelFromSinglePhoto(json_fn):
    angle, milieu_x, milieu_y, intersection, dist_x, dist_y = executePiplineForInformation(json_fn)

    # print("the angle " + str(angle))
    # print("dist_x " + str(dist_x))
    # print("dist_y " + str(dist_y))

    classe = "Error"

    try:
        imageName = ntpath.basename(json_fn).replace(".json", ".jpg")
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

        # if (angle >= 0 and angle < 22.5 or angle >= 337.5 and angle <= 360) or (angle >= 157.5 and angle < 202.5):
        # chance to be center center!!
        # if dist_x <= 29:
        #  classe = "MiddleCenter"
        # print("Classe ==>" + classe)
        # print("imageName ==>" + imageName)
        # print("json_fn ==>" + json_fn)
    except Exception as e:
        print(str(e))
        sys.exit(0)
    return classe


if __name__ == '__main__':
    # trial()
    json_fn = "/Users/hazem.fahmy/gazedetectionandanalysisdnn/60652.json"
    img = cv2.imread("/Users/hazem.fahmy/gazedetectionandanalysisdnn/60652.jpg")
    data_file = open(json_fn)
    data = json.load(data_file)
    testdraw(data, img)
    s
    # read vars
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Put path the location of images to classify
    images = "images"
    images = readStringVar(config, "myvars", "images")
    json_fns = glob(images + "/*.json")
    classes = ['BottomCenter', 'BottomLeft', 'BottomRight', 'MiddleLeft', 'MiddleRight', 'TopCenter',
               'TopLeft', 'TopRight']

    for c in classes:
        if not isdir("outputs/" + c):
            try:
                os.mkdir("outputs/" + c)
            except OSError:
                print("Creation of the directory %s failed" % "outputs/" + c)
            else:
                print("Successfully created the directory %s " % "outputs/" + c)

    for json_fn in json_fns:
        print(json_fn)

        exists = isfile("%s.jpg" % json_fn[:-5])
        if not exists:
            print("Image does not exist for the json file: " + json_fn)
            continue

        if CLASSIFICATION_METHOD == CLASSIFY_WITH_YAW_PICTH:
            classe = getLabelFromSinglePhotoYawPitch(json_fn)
        else:
            classe = getLabelFromSinglePhoto(json_fn)

        imageName = ntpath.basename(json_fn).replace(".json", ".jpg")
        if (classe != "Error"):
            shutil.copyfile(images + "/" + imageName, "outputs/" + classe + "/" + imageName)
        # if(debug ==  True):
        # cv2.waitKey(200)
        # input('Press key ')



def hmcrop(net, orig_dir, new_dir, mode, file):
    result, outputProb, layerHMa, fileName, heatmap_Alex = dnn.classifyOneImage(file, net, orig_dir, new_dir, 0, 1, mode)
    crop = Image.open(str(new_dir + "/" + fileName + "_crop.jpg"))
    crop = np.array(crop)
    print(result)
    total = int(256*256)
    k = 0.9
    r = 1-k

    for i in range(0,int(k*total)):
        coord = np.unravel_index(np.argmin(heatmap_Alex, axis=None), heatmap_Alex.shape)
        heatmap_Alex[coord[0]][coord[1]] = [255, 255, 255]
        crop[coord[0]][coord[1]] = [0, 0, 0]
    #for i in range(0, int(r*total)):
    #    coord = np.unravel_index(np.argmin(heatmap_Alex, axis=None), heatmap_Alex.shape)
    #    heatmap_Alex[coord[0]][coord[1]] = [255, 255, 255]

    Image.fromarray(crop, 'RGB').save(new_dir + "/" + str(fileName) + "_flip.jpg")
    result, outputProb, layerHMa, fileName, heatmap_Alex = dnn.classifyOneImage(str(fileName) + "_flip.jpg", net, new_dir, new_dir, 12, 0, mode)
    print(result)

def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
 px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
   (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
 py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
   (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
 return [px, py]

def process_json_list(json_list,img):
 ldmks = [eval(s) for s in json_list]
 return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

def draw_gaze(data, img, path, fileName):
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    #for ldmk in np.vstack([ldmks_interior_margin[0], ldmks_interior_margin[4], ldmks_interior_margin[12],
    #                       ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)]]):
    #    cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
    x1 = int(ldmks_interior_margin[0][0])
    y1 = int(ldmks_interior_margin[0][1])
    x2 = int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0])
    y2 = int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1])
    x3 = int(ldmks_interior_margin[4][0])
    y3 = int(ldmks_interior_margin[4][1])
    x4 = int(ldmks_interior_margin[12][0])
    y4 = int(ldmks_interior_margin[12][1])

    intersection = findIntersection(x1, y1, x2, y2, x3, y3, x4, y4)

    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    ldmks_caruncle = process_json_list(data['caruncle_2d'], img)
    ldmks_iris = process_json_list(data['iris_2d'], img)

    # Draw black background points and lines
    # for ldmk in np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]]):
    #    cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (0, 0, 0), -1)
    # cv2.polylines(img, np.array([ldmks_interior_margin[:, :2]], int), True, (0, 0, 0), 2)
    # cv2.polylines(img, np.array([ldmks_iris[:, :2]], int), True, (0, 0, 0), 2)

    # Draw green foreground points and lines
    #cv2.polylines(img, np.array([ldmks_interior_margin[:, :2]], int), True, (0, 255, 0), 1)
    #cv2.polylines(img, np.array([ldmks_iris[:, :2]], int), True, (0, 255, 0), 1)

    look_vec = list(eval(data['eye_details']['look_vec']))

    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    look_vec[1] = -look_vec[1]
    cv2.line(img, tuple(eye_c), tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int)), (0, 0, 0), 3)
    cv2.line(img, tuple(eye_c), tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int)), (0, 255, 255), 3)
    # cv2.imshow("syntheseyes_img", img)
    # cv2.imwrite("annotated_%s.png"%json_fn[:-5], img)
    # cv2.waitKey(1)
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))
    # cv2.line(img, point_A, point_B, (0, 0, 0), 3)
    # cv2.line(img, point_A, point_B, (0, 255, 255), 2)
    iris = ldmks_iris
    #print(iris)
    #print(len(iris))
    #for ldmk in np.vstack([ldmks_iris[::2]]):

    #cv2.circle(img, (int(iris[0][0]), int(iris[0][1])), 2, (0, 255, 0), -1)
    #cv2.circle(img, (int(iris[1][0]), int(iris[1][1])), 2, (0, 255, 0), -1)
    #cv2.circle(img, (int(iris[2][0]), int(iris[2][1])), 2, (0, 255, 0), -1)
    cv2.circle(img, (int(iris[0][0]), int(iris[0][1])), 2, (0, 255, 0), -1)

    ldmk1 = ldmks_interior_margin[4]
    ldmk2 = ldmks_interior_margin[12]
    x1 = int(ldmk1[0])
    y1 = int(ldmk1[1])
    x2 = int(iris[0][0])
    y2 = int(iris[0][1])
    cv2.circle(img, (int(x1), int(y1)), 2, (0, 255, 0), -1)
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #print(dist)
    #print(y2 - y1)
    #print(y1 - y2)

    #if(y2 - y1 < 0):
    #    print("uncommon")
    angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
                                                                                      point_B[1] - point_A[1])
    angle = (angle * 180) / math.pi
    while (angle < 0):
        angle = angle + 360
    #print(angle)

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, str(angle), (70, 400), font, 1, (255, 255, 255), 2)
    cv2.imwrite(path + "/" + fileName + ".jpg", img)

    # drawCircles(img, intersection, milieu(x1, y1, x2, y2), milieu(x3, y3, x4, y4))
    # cv2.imshow('ImageWindow', img)
    # cv2.waitKey(10000)


def getclass(file, img_dir):
    fileName = str(file).split(".")[0]
    img = cv2.imread(str(img_dir) + "/" + fileName + ".jpg")
    json_fn = img_dir + "/" + fileName + ".json"
    data_file = open(json_fn)
    data = json.load(data_file)

    head_pose = data['head_pose']
    hp = head_pose
    hp1 = float(hp.split(",")[0].split("(")[1])
    hp2 = float(hp.split(", ")[1].split(",")[0])
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    ldmk1 = ldmks_interior_margin[4]
    ldmk2 = ldmks_interior_margin[12]
    x1 = int(ldmk1[0])
    y1 = int(ldmk1[1])
    x2 = int(ldmk2[0])
    y2 = int(ldmk2[1])

    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist = int(dist)
    if (dist < 20):
        classe = "Closed"
    else:
        classe = "Opened"

    return classe
def do_label_save(AS, img_dir, json_dir, new_dir):
    i = 1
    for file in AS:
        if(file == 0):
            file = file
        else:
                fileName = str(file).split(".")[0]
                img = cv2.imread(str(img_dir) + "/" + fileName + ".jpg")
                json_fn = json_dir + "/" + fileName + ".json"
                data_file = open(json_fn)
                data = json.load(data_file)

                head_pose = data['head_pose']
                hp = head_pose
                hp1 = float(hp.split(",")[0].split("(")[1])
                hp2 = float(hp.split(", ")[1].split(",")[0])
                ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
                ldmk1 = ldmks_interior_margin[4]
                ldmk2 = ldmks_interior_margin[12]
                x1 = int(ldmk1[0])
                y1 = int(ldmk1[1])
                x2 = int(ldmk2[0])
                y2 = int(ldmk2[1])

                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dist = int(dist)
                if(dist<20):
                    classe="Closed"
                else:
                    classe="Opened"
                #angle, point_A, point_B, point_C = lable.computeAngle(data, img)
                #if (hp1 >= 340 and hp1 <= 359.9) and (hp2 >=160 and hp2 <190):
                #    classe = "1"
                #if (hp1 >= 340 and hp1 <= 359.9) and (hp2 >=190 and hp2 <220):
                #    classe = "2"
                #if (hp1 >= 0 and hp1 <= 20) and (hp2 >=160 and hp2 <190):
                #    classe = "3"
                #if (hp1 >= 0 and hp1 <= 20) and (hp2 >=190 and hp2 <220):
                #    classe = "4"
                #if angle > 157.5 and angle < 202.5:
                #    classe = "MiddleRight"
                #if angle > 202.5 and angle < 247.5:
                #    classe = "BottomRight"
                #if angle > 247.5 and angle < 292.5:
                #    classe = "BottomCenter"
                #if angle > 292.5 and angle < 337.5:
                #    classe = "BottomLeft"
                #if angle >= 337.5:
                #    classe = "MiddleLeft"
                save_dir = new_dir + "/" + classe + "/"
                if not exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(str(save_dir) + str(fileName) + ".jpg", img)
                print("labelled and saved " + str(i) + " images")
                i = i + 1

def detect_strange(json_fn, img, threshold):
    data_file = open(json_fn)
    data = json.load(data_file)
    interior = process_json_list(data['interior_margin_2d'], img)
    caruncle = process_json_list(data['caruncle_2d'], img)
    iris = process_json_list(data['iris_2d'], img)

    ldmk1 = interior[4]
    ldmk2 = interior[12]
    upper_x = int(ldmk1[0])
    upper_y = int(ldmk1[1])
    left_x = int(iris[0][0])
    left_y = int(iris[0][1])
    right_x = int(iris[17][0])
    right_y = int(iris[17][1])
    lower_x = int(ldmk2[0])
    lower_y = int(ldmk2[1])
    top_x = int(iris[10][0])
    top_y = int(iris[10][1])
    bot_x = int(iris[23][0])
    bot_y = int(iris[23][1])
    milieu_x = getMiddelX(data, img)
    angle, point_A, point_B, point_C = computeAngle(data, img)
    dist_x = getDistBetweenTwoPoints(point_A, milieu_x)
    #cv2.circle(img, (lower_x, lower_y), 2, (0, 255, 0), -1)
    #cv2.circle(img, (bot_x, bot_y), 2, (0, 255, 0), -1)

    #dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dif = [0] * 2
    if(lower_y - bot_y < threshold):
        state ='bottom uncommon'
        state = 1
    elif(top_y - upper_y < threshold):
        state ='top uncommon'
        state = 1
    else:
        state = 'common'
        state = 0
    dif[0] = lower_y - bot_y
    dif[1] = top_y - upper_y
    return dif[0], dif[1], state
    #if(dist_x <= 29):
    #    state = state + 'and closed iris'
    #font = cv2.FONT_HERSHEY_COMPLEX
    #cv2.putText(img, str(state), (70, 400), font, 1, (255, 255, 255), 2)
    #cv2.imwrite(path + "/" + fileName + "_Gaze.jpg", img)


def getDistBetweenTwoPoints(point_A, milieu_x):
    return math.sqrt((point_A[0] - milieu_x[0]) * (point_A[0] - milieu_x[0]) + (point_A[1] - milieu_x[1]) * (
            point_A [1] - milieu_x[1]))


def computeAngle(data,img):
    ldmks_iris = process_json_list(data['iris_2d'],img)
    look_vec = list(eval(data['eye_details']['look_vec']))
    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    look_vec[1] = -look_vec[1]
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))
    angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
                                                                                      point_B[1] - point_A[1]);
    angle = (angle * 180) / math.pi

    while (angle < 0):
        angle = angle + 360
    return angle, point_A, point_B, point_C


def getMiddelX(data, img):
    ldmks_interior_margin = process_json_list(data['interior_margin_2d'], img)
    return [int(ldmks_interior_margin[0][0]), int(ldmks_interior_margin[0][1]),
                      int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0]),
                      int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1])]

