import random

from imports import basename, join, dirname, os, exists, pd, shutil
import wand
from wand.image import Image as NImage
from PIL import Image as BImage
from PIL import ImageFilter
from testModule import testModelForImg

def inject(caseFile, csv, faults, classes, destPath, n=10):
    print(f"injecting {n} faults per class")
    datasetsPath = join(str(caseFile["filesPath"]), "DataSets")
    #faultsPath = join(datasetsPath, "BN_TestSet")
    backupPath = join(datasetsPath, "Backup")
    imageList = pd.read_csv(csv)
    # noise - blur - hands - hands with shirt - masks - sunglasses - eyeglasses - non injected fault
    print("faults per class", classes)
    print("total faults", faults)
    for index, row in imageList.iterrows():
        fileName = str(basename(row["image"]))
        backupDir = join(backupPath, row["expected"])
        ensure_dir(backupDir)
        if row["result"] == "Correct":
            faultFlag = False
            for fault in faults:
                if fileName.startswith(fault):
                    faultFlag = True
            if not faultFlag:
                if classes[row["expected"]]['N'] < n:
                    if addNoise(caseFile, row, backupDir, join(destPath, row["expected"])):
                        faults['N'] += 1
                        classes[row["expected"]]['N'] += 1
                if classes[row["expected"]]['B'] < n:
                    if addBlur(caseFile, row, backupDir, join(destPath, row["expected"])):
                        faults['B'] += 1
                        classes[row["expected"]]['B'] += 1

        print(index, "faults per class", classes, end="\r")
        breakFlag = True
        for label in classes:
            for fault in faults:
                if classes[label][fault] < n:
                    breakFlag = False
        if breakFlag:
            break
                #shutil.move(row["image"], join(backupDir, basename(row["image"])))
    shutil.rmtree(backupPath)
    print("done injecting")
    return classes, faults

def setClassesFaults(caseFile):
    if caseFile["datasetName"] == "GD" or caseFile["datasetName"] == "OC":
        faults = {'N': 0, 'B': 0, 'NA': 0}  # GD / OC
    else:   # HPD / FLD
        faults = {'N': 0, 'B': 0, 'H': 0, 'HS': 0, 'M': 0, 'S': 0, 'E': 0, 'NA': 0}
    classes = {}
    if caseFile["datasetName"] != "FLD":
        classList = os.listdir(caseFile["testDataPath"])
        for label in classList:
            if not label.startswith("."):
                classes[str(label)] = {}
                for fault in faults:
                    classes[str(label)][fault] = 0
    return classes, faults

def getFaults(csv, classes, faults):
    imageList = pd.read_csv(csv)
    for index, row in imageList.iterrows():
        if row["result"] == "Wrong":
            faultFlag = False
            for fault in faults:
                if str(basename(row["image"])).startswith(fault):
                    faults[fault] += 1
                    classes[row["expected"]][fault] += 1
                    faultFlag = True
            if not faultFlag:
                faults['NA'] += 1
                classes[row["expected"]]['NA'] += 1
    return classes, faults


def bagFaults(csv, classes, faults, newDir, bag):
    print("bagging")
    imageList = pd.read_csv(csv)
    faultImages = []
    for index, row in imageList.iterrows():
        if row["result"] == "Wrong":
            for fault in faults:
                if basename(row["image"]).startswith(fault):
                    faultImages.append(row["image"])

    for label in classes:
        for fault in faults:
            while 0 < classes[label][fault] < bag:
                imagePath = faultImages[random.randint(0, len(faultImages)-1)]
                imgClass = basename(dirname(imagePath))
                imageName = basename(imagePath)
                if imgClass == label and imageName.startswith(fault):
                    imgExt = "." + imageName.split(".")[1]
                    newName = imageName.split(imgExt)[0] + "_" + str(classes[label][fault]) + imgExt
                    if not exists(join(newDir, basename(dirname(imagePath)), newName)):
                        shutil.copy(imagePath, join(newDir, basename(dirname(imagePath)), newName))
                        classes[label][fault] += 1
                        faults[fault] += 1
        print(label, classes[label])
    return classes, faults


def addNoise(caseFile, row, backupDir, newDir):
    img = NImage(filename=row["image"])
    img.noise("laplacian", attenuate=5.0)
    if not exists(join(backupDir, "N" + basename(row["image"]))):
        img.save(filename=join(backupDir, "N" + basename(row["image"])))
        DNNResult, _ = testModelForImg(caseFile["DNN"], row["expected"], join(backupDir, "N" + basename(row["image"])), caseFile)
        if not DNNResult:
            shutil.copy(join(backupDir, "N" + basename(row["image"])), join(newDir, "N" + basename(row["image"])))
            return True
        else:
            return False
    else:
        return False


def addBlur(caseFile, row, backupDir, newDir):
    img = BImage.open(row["image"])
    img = img.filter(ImageFilter.BoxBlur(3))
    if not exists(join(backupDir, "B" + basename(row["image"]))):
        img.save(join(backupDir, "B" + basename(row["image"])))
        DNNResult, _ = testModelForImg(caseFile["DNN"], row["expected"], join(backupDir, "B" + basename(row["image"])), caseFile)
        if not DNNResult:
            shutil.copy(join(backupDir, "B" + basename(row["image"])), join(newDir, "B" + basename(row["image"])))
            return True
        else:
            return False
    else:
        return False


def genSunGlasses():
    return


def genEyeGlasses():
    return


def genMasks():
    return

def ensure_dir(dir):
    if not exists(dir):
        os.makedirs(dir)
