from imports import np, random, makedirs, torch, pd, join, basename, exists, isfile, sys, os, cv2, dlib, dirname, \
    imageio, subprocess, shutil, math, random, distance, math, time, transforms, Variable
#import torchvision.datasets.vision
from HeatmapModule import generateHeatMap, doDistance
from assignModule import getClusterData
from testModule import testModelForImg
from dnnModels import ConvModel
import ieedatavendor as ieeDV
#import mosa
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.model.population import Population
from pymoo.model.individual import Individual

import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
import dataSupplier as DS
import nsga2P0
import nsga2P1
import nsga2plus
import pickle
import matplotlib.pyplot as plt

# components = ["mouth", "noseridge", "nose", "rightbrow", "righteye", "lefteye", "leftbrow"]
components = ["mouth"]
#outputPath = "/Users/hazem.fahmy/Documents/HPD/"
# outputPath = "/home/users/hfahmy/DEEP/HPC/HPD/"
outputPath = "/Users/android/Documents/HPD/"
#DIR = "/Users/hazem.fahmy/Documents/HPC/HUDD/runPy/"
# DIR = "/home/users/hfahmy/DEEP/HPC/HUDD/runPy/"
DIR = "/Users/android/Documents/HPC/HUDD/runPy/"

blenderPath = "/Applications/Blender.app/Contents/MacOS/blender"
# blenderPath = "/home/users/hfahmy/blender-2.79/blender"
path = join(outputPath, "IEEPackage")
layer = 12
#nVar = 20
nVar = 13
indvdSize = 1
BL = False
width = 752 // 2  # 376
height = 480 // 2  # 240
counter = 0


def search(caseFile, clsData, centroidHM, clusterRadius, popSoze, nGen):
    global layer
    global prevPOP
    layer = int(caseFile["selectedLayer"].replace("Layer", ""))
    outPath = join(caseFile["filesPath"], "Pool")
    caseFile["SimDataPath"] = outPath
    if not exists(outPath):
        makedirs(outPath)
    r = random.randint(1, 1000)
    print(r)
    #GI_path = join(caseFile["filesPath"], "GeneratedImages_SEDE25_test")
    GI_path = join(caseFile["filesPath"], "GeneratedImages_SEDE25")
    #GI_path = join(caseFile["filesPath"], "GeneratedImages_NSGA")

    if not exists(GI_path):
        makedirs(GI_path)
    dp = join(GI_path, "details" + str(r) + ".txt")
    open(dp, "w")
    #dall = None
    #dall_ind = None
    dall_ind_cluster = None
    for ID in [5]:
    #torch.save(caseFile["DNN"], caseFile["modelPath"])
    #exit(0)
    #for ID in clsData:
        # for ID in [6]:
        # for ID in [3, 6]:
        # for ID in [3]:
        # for ID in [10, 11, 12, 13, 14, 16, 17]:
    #for ID in clsData['clusters']:
        #if exists(join(GI_path, str(ID))):
        #    continue
        #if ID == 3 or ID == 7:
        #    continue
        file = open(dp, "a")
        file.write("Cluster:" + str(ID) + "\n")
        print("Searching images for 5Cluster:", ID)
        # evalImages(clsData, caseFile)
        caseFile["CR"] = clusterRadius[ID]
        caseFile["CH"] = centroidHM[ID]
        caseFile["ID"] = ID
        caseFile["GI"] = GI_path
        caseFile["xl"] = setX(1, "L")
        caseFile["xu"] = setX(1, "U")
        popSize = [500, 15, 15]
        # popSize = [30, 3, 3]
        nGen = [3000, 5, 10]
        # nGen = [15, 3, 75]
        # nGen = [2, 2, 2]
        file.write("PopSize:" + str(popSize) + "\n")
        file.write("Iterations:" + str(nGen) + "\n")
        start1 = time.time()
        initpop0, ent0 = doProblem(popSize[0], popSize[0], nGen[0], caseFile, None, None, 1)
        file.write(doTime("Problem 1 Cost:", start1) + "\n")
        start1 = time.time()


        #np.save(join(caseFile["GI"], str(caseFile["ID"]),"2.npy"), problem.dhm)

        continue
        initpop1, ent1 = doProblem(popSize[1], popSize[1], nGen[1], caseFile, initpop0, ent0, 2)
        file.write(doTime("Problem 2 Cost:", start1) + "\n")
        start = time.time()
        initpop2, ent2 = doProblem(popSize[2], popSize[2], nGen[2], caseFile, initpop1, ent1, 3)
        file.write(doTime("Problem 3 Cost:", start) + "\n")
        start = time.time()
        #_, _ = doProblem(popSize[3], popSize[3], nGen[3], caseFile, initpop2, ent2, 4)
        #file.write(doTime("Problem 3 Cost:", start) + "\n")
        file.write(doTime("Total Cost:", start1) + "\n")
        file.close()


def do_analysis(caseFile, clusters):
    dall = None
    dall_ind = None
    for ID in clusters:
        dsim = np.load(join(caseFile["filesPath"], "GeneratedImages_SEDE25", str(ID), "1.npy"))
        dind = np.load(join(caseFile["filesPath"], "GeneratedImages_SEDE25", str(ID), "individuals.npy"))
        if dall is None:
            dall = {}
            dall_ind = {}
            for i in range(0, len(dsim)):
                print(i)
                dall[str(i)] = [dsim[i]]
                dall_ind[str(i)] = [dind[i]]
        else:
            for i in range(0, 40):
                dall[str(i)].append(dsim[i])
                dall_ind[str(i)].append(dind[i])
    c = []
    davg1 = []
    dmax = []
    dmin = []
    j = 1
    for i in range(0, 40):
        c.append(j)
        davg1.append(sum(dall[str(i)])/len(dall[str(i)]))
        dmax.append(max(dall[str(i)]))
        dmin.append(min(dall[str(i)]))
        print(i+1, sum(dall[str(i)])/len(dall[str(i)]))
        j += 1
    plt.plot(c, dmax, "g:")
    plt.plot(c, davg1)
    plt.plot(c, dmin, "r:")
    plt.legend(['Max.', 'Avg.', 'Min.'])

    plt.ylabel("Avg. Sim. Params. Dists.")
    plt.xlabel("execution time (hrs)")
    plt.savefig(join(caseFile["GI"],"all_.png"))
    plt.cla()
    plt.clf()

    davg2 = []
    dmax2 = []
    dmin2 = []
    for i in range(0, 40):
        #c.append(i+1)
        davg2.append(sum(dall_ind[str(i)])/len(dall_ind[str(i)]))
        dmax2.append(max(dall_ind[str(i)]))
        dmin2.append(min(dall_ind[str(i)]))
        print(i+1, sum(dall_ind[str(i)])/len(dall_ind[str(i)]))
    plt.plot(c, dmax2, "g:")
    plt.plot(c, davg2)
    plt.plot(c, dmin2, "r:")
    plt.legend(['Max.', 'Avg.', 'Min.'])

    plt.ylabel("% of cluster members")
    plt.xlabel("execution time (hrs)")
    plt.savefig(join(caseFile["GI"],"all_individduals.png"))
    plt.cla()
    plt.clf()
    GI_path = join(caseFile["filesPath"], "GeneratedImages_NSGA")

    if not exists(GI_path):
        makedirs(GI_path)
    dp = join(GI_path, "details" + str(r) + ".txt")
    open(dp, "w")
    dall2 = None
    for ID in clusters:

        if ID == 7 or ID == 3 or ID == 9:
            continue
        dsim = np.load(join(GI_path, str(ID), "1.npy"))
        dind = np.load(join(GI_path, str(ID), "individuals.npy"))
        if dall2 is None:
            dall2 = {}
            dall3 = {}
            for i in range(0, len(dsim)):
                dall2[str(i)] = [dsim[i]]
                #dall3[str(i)] = [dind[i]]
        else:
            for i in range(0, len(dsim)):
                dall2[str(i)].append(dsim[i])
                #dall3[str(i)].append(dind[i])
    c = []
    davg2 = []
    dmax = []
    dmin = []
    for i in range(6, len(dsim)):
        c.append(i)
        print(dall[str(i)])
        print(dall2[str(i)])
        print("NSGA, AVG Params", sum(dall[str(i)])/len(dall[str(i)]))
        print("NSGA, indiv.", davg2[i])
        print("NSGA, AVG indiv.", davg2[i])
        print("SEDE, AVG", sum(dall2[str(i)])/len(dall2[str(i)]))
        print("SEDE, AVG indiv.", davg2[i])
        print(i, stats.mannwhitneyu(dall[str(i)], dall2[str(i)]))

def doProblem(pop_size, popS_new, n_gen, caseFile, prev_pop, prev_en, probNum):
    dirPath = join(caseFile["GI"], str(caseFile["ID"]))
    if not exists(dirPath):
        makedirs(dirPath)
    CP = join(caseFile["GI"], str(caseFile["ID"]), "PF" + str(probNum) + ".pop")
    init_ = [get_sampling("real_random"), prev_pop, prev_pop]
    # init_ = [get_sampling("real_random"), get_sampling("real_random"), prev_pop]
    out_ = ['w', 'w', 'a']
    noff_ = [1, 1, None]
    csv = [False, True, True]
    res = solveProblem(pop_size=pop_size, n_gen=n_gen, n_offsprings=noff_[probNum - 1], sampling=init_[probNum - 1],
                                          caseFile=caseFile, probNum=probNum, initpop=prev_pop, ent=prev_en)
                       #caseFile=caseFile, probNum=probNum - 1, initpop=prev_pop, ent=prev_en)
    # PF = getPF(res.pop, CP)
    print(res)
    if hasattr(res, 'pop'):
        res = res.pop
    exportResults_2(res, "/PF/" + str(probNum) + ".txt", "/PF/" + str(probNum), probNum, dirPath,
    #exportResults_2(res.pop, "/PF/" + str(probNum - 1) + ".txt", "/PF/" + str(probNum - 1), probNum - 1, dirPath,
                    open(dirPath + "/results.csv", out_[probNum - 1]), caseFile, prev_en, prev_pop, csv[probNum - 1])

    if probNum != 3:
        new_pop, new_en = prepareINIT(res, caseFile, dirPath, probNum + 1, popS_new)
        #new_pop, new_en = prepareINIT(res.pop, caseFile, dirPath, probNum, popS_new)
        for member in new_pop:
            print("F", min(member.F))
        return new_pop, new_en
    else:
        return None, None

import pymoo
def solveProblem(pop_size, n_gen, n_offsprings, sampling, caseFile, probNum, ent, initpop):
    problem = HUDDProblem_N(caseFile, probNum, ent, initpop, n_gen, pop_size)
    CP = join(caseFile["GI"], str(caseFile["ID"]), "res" + str(probNum) + ".pop")
    CP_ = join(caseFile["GI"], str(caseFile["ID"]), "init" + str(probNum+1) + ".pop")
    CP2 = join(caseFile["GI"], str(caseFile["ID"]), "res" + str(probNum) + ".ckpt.npy")
    print("solving problem", probNum)
    #if probNum == 0:
    #    algorithm = nsga2P0.NSGA2P0(pop_size=pop_size, sampling=sampling,
    #                                crossover=get_crossover("real_sbx", prob=0.7, eta=20),
    #                                mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    if probNum == 1:
        algorithm = nsga2P1.NSGA2P1(pop_size=pop_size, sampling=sampling,
                                    crossover=get_crossover("real_sbx", prob=0.3, eta=20),
                                    mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)

        #algorithm = NSGA2(pop_size=pop_size, sampling=sampling,
        #                            crossover=get_crossover("real_sbx", prob=0.3, eta=20),
        #                            mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    elif probNum == 2:
        algorithm = nsga2plus.NSGA2Plus(pop_size=pop_size, sampling=sampling,
                                        crossover=get_crossover("real_sbx", prob=0.3, eta=20),
                                        mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    # algorithm = NSGA3(ref_dirs=get_reference_directions("energy", pop_size, pop_size, seed=1), pop_size=pop_size,
    #                  sampling=sampling, crossover=get_crossover("real_sbx", prob=0.7, eta=20),
    #                  mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    elif probNum == 3:
        algorithm = nsga2plus.NSGA2Plus(pop_size=pop_size, sampling=sampling,
                                        crossover=get_crossover("real_sbx", prob=0.3, eta=20),
                                        mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
        # algorithm = NSGA3(ref_dirs=get_reference_directions("energy", pop_size, pop_size, seed=1), pop_size=pop_size,
        #                  sampling=sampling, crossover=get_crossover("real_sbx", prob=0.7, eta=20),
        #                  mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    if isfile(CP):
        res = loadCP(CP)
        print("Loaded Checkpoint:", res)
    else:
        res = minimize(problem, algorithm, ('n_gen', n_gen), n_offsprings=n_offsprings, verbose=True)
        if probNum == 1:
            print(problem.dsim, problem.c)
            np.save(join(caseFile["GI"], str(caseFile["ID"]),"1.npy"), problem.dsim)
            np.save(join(caseFile["GI"], str(caseFile["ID"]),"2.npy"), problem.dhm)
            np.save(join(caseFile["GI"], str(caseFile["ID"]),"individuals.npy"), problem.dind)
            plt.plot(problem.c, problem.dsim)
            plt.ylabel("Avg. Sim. Params. Dists.")
            plt.xlabel("execution time (hrs)")
            plt.savefig(join(caseFile["GI"], str(caseFile["ID"]),"1.png"))
            plt.cla()
            plt.clf()


            plt.plot(problem.c, problem.dhm)

            plt.ylabel("Avg. Heatmaps Dists.")
            plt.xlabel("execution time (hrs)")
            plt.savefig(join(caseFile["GI"], str(caseFile["ID"]),"2.png"))
            plt.cla()
            plt.clf()


            plt.plot(problem.c, problem.dind)

            plt.ylabel("% of cluster members")
            plt.xlabel("execution time (hrs)")
            plt.savefig(join(caseFile["GI"], str(caseFile["ID"]),"individuals.png"))
            plt.cla()
            plt.clf()
            #plt.plot(problem.dhm, problem.c)

        saveCP(res, CP)

    print(res)

    if hasattr(res, 'pop'):
        res = res.pop
    #for member in res:
    #    print(min(member.F))
    exportResults_2(res, "/Pop/" + str(probNum) + ".txt", "/Pop/" + str(probNum), probNum,
                    join(caseFile["GI"], str(caseFile["ID"])), None,
                    caseFile, ent, initpop, False)

    return res


class HUDDProblem_N(Problem):
    def __init__(self, caseFile, probID, prevEN, prevPOP, n_gen, pop_size, **kwargs):
        self.outPath = caseFile["outputPath"]
        self.problemDict = {}
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.clusterRadius = caseFile["CR"]
        self.centroidHM = caseFile["CH"]
        self.caseFile = caseFile
        self.probID = probID
        self.prevEN = prevEN
        self.prevPOP = prevPOP
        self.previousPop = prevPOP
        self.t = time.time()
        self.dsim = []
        self.dhm = []
        self.dind = []
        self.c = []
        self.counter = 0
        self.flag = True
        if probID == 1:
       #     n_obj = 2
       # elif probID == 0:
            n_obj = 1
        else:
            n_obj = 25

        xl = setX(1, "L")
        xu = setX(1, "U")
        self.xl = xl
        self.xu = xu
        print("Solving Problem .. ", probID)
        super().__init__(n_var=nVar, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=False, **kwargs,
                         type_var=np.float)

    def _evaluate(self, X, out, *args, **kwargs):  # Batchwise
        start1 = time.time()
        if not exists(self.outPath):
            makedirs(self.outPath)
        if self.probID == 1:
            # print("prevPop", self.previousPop)
            F = HUDDevaluate_Pop1(X, self.caseFile, self.previousPop)
            #F = HUDDevaluate_Pop1N(X, self.caseFile, self.previousPop)
        elif self.probID == 2:
            F = HUDDevaluate_Pop2(X, self.caseFile, self.previousPop)
        elif self.probID == 3:
            F = HUDDevaluate_Pop3(X, self.caseFile, self.prevPOP)
        elif self.probID == 0:
            F = HUDDevaluate_Pop1(X, self.caseFile, self.previousPop)
        if out is None:
            return F
        else:
            out["F"] = np.array(F)
        #if len(F[0]) > 1:
        #    for n in F:
        #        print(min(n))
        #else:
        print("F", F)
        if self.flag:
            self.individual_analysis()
        print(doTime("Iteration Cost:", start1))
        #self.counter += 1
    def individual_analysis(self):
        d_sim = []
        d_hm = []
        d_ind = []
        if hasattr(self, "pop"):
            # if self.previousPop is not None:
            self.previousPop = self.pop
            # print("Comparing Current vs. Prev., F=1-2ndClosest")
            for x in self.previousPop:

                #imgPath, F, new_img, new_label = generateAnImage(x.X, self.caseFile)
                #imgPath += ".png"
                #print(x.F[0])
                #if F:

                #    N, DNNResult, P, L, D3, HM = doImage(imgPath, self.caseFile, self.caseFile["CH"],
                #                                         [new_img, new_label])
                    #if D3 / self.caseFile["CR"] <= 1:
                if x.F[0] <= 1:
                        #d_ind.append(D3 / self.caseFile["CR"])
                        d_ind.append(x.F[0])
                        for x3 in self.previousPop:
                            #imgPath, F3, new_img, new_label = generateAnImage(x3.X, self.caseFile)
                            #imgPath += ".png"
                            #if F3:
                            if x3.F[0]:
                            #    N, DNNResult, P, L, D3, _ = doImage(imgPath, self.caseFile, HM[layer],
                            #                                        [new_img, new_label])
                                #f3 = D3 / self.caseFile["CR"]
                                f3 = x3.F[0]
                                fp = doParamDist(x.X, x3.X, self.xl, self.xu)

                                if fp != 0 and f3 <= 1:
                                    d_sim.append(fp)
                                #if D3 != 0:
                                #    d_hm.append(D3)
        if len(d_sim) > 0:
            dsim_avg = sum(d_sim) / len(d_sim)
        else:
            dsim_avg = 0
        if len(d_hm) > 0:
            dhm_avg = sum(d_hm) / len(d_hm)
        else:
            dhm_avg = 0
        self.counter += 1
        self.dsim.append(dsim_avg)
        self.dhm.append(dhm_avg)
        self.c.append(self.counter)
        self.dind.append(100 * (len(d_ind) / 10))

        print(d_ind, self.dind[-1], d_sim, d_hm, dsim_avg, dhm_avg, self.counter)
    def _evaluate_EW(self, x, out, *args, **kwargs):  # Elementwise
        self.counter += 1
        print(str(100 * (self.counter / (self.n_gen * self.pop_size)))[0:5] + "%", end="\r")
        if not exists(self.outPath):
            makedirs(self.outPath)
        if self.probID == 1:
            problemDict = HUDDevaluate_N(x, self.caseFile, self.prevEN, self.prevPOP)
            self.problemDict[processX(x)] = problemDict
            out["F"] = np.array([problemDict["F1"]])
        elif self.probID == 0:
            problemDict = HUDDevaluate_N(x, self.caseFile, None, None)
            self.problemDict[processX(x)] = problemDict
            out["F"] = np.array([problemDict["F1"]])
        else:
            F_list = []
            for i in range(0, len(self.prevPOP)):
                problemDict = HUDDevaluate_N(x, self.caseFile, self.prevEN, self.prevPOP[i].X)
                self.problemDict[processX(x)] = problemDict
                if self.probID == 2:
                    F_list.append(problemDict["FY"])
                    out["F"] = np.array([problemDict["FY"]])
                elif self.probID == 3:
                    F_list.append(problemDict["FY_"])
                    out["F"] = np.array([problemDict["FY_"]])
            out["F"] = np.array(F_list)
        print(out["F"])
        # if time.time() - self.t > 2h:
        # terminate


def HUDDevaluate_Pop1N(X, caseFile, Y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    d_sim = []
    d_hm = []
    for x in X:
        if x is None:
            fy.append(math.inf)
            continue

        imgPath, F, _ = generateAnImage(x, caseFile)
        imgPath += ".png"
        dists = []
        fy_ = math.inf
        f1 = math.inf
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)
            f1 = D / CR
        f1_.append(f1)
        fy.append([fy_])
    # print("F1:", f1_)

    return f1_
def HUDDevaluate_Pop1(X, caseFile, Y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    d_sim = []
    d_hm = []
    for x in X:
        if x is None:
            fy.append(math.inf)
            continue

        imgPath, F, new_img, new_label = generateAnImage(x, caseFile)
        imgPath += ".png"
        dists = []
        fy_ = math.inf
        f1 = math.inf
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH, [new_img, new_label])
            f1 = D / CR
            if f1 <= 1:
                # Compare current vs. previous population, get closest and 2nd closest
                if Y is not None:
                    # print("Comparing Current vs. Prev., F=1-2ndClosest")
                    for x2 in Y:
                        imgPath2, F2, new_img, new_label = generateAnImage(x2.X, caseFile)
                        imgPath2 += ".png"
                        if x2.get("F")[0] <= 1:
                            N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH, [new_img, new_label])
                            _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer], [new_img, new_label])
                            fp = doParamDist(x, x2.X, xl, xu)
                            dists.append(fp)
                        else:
                            dists.append(math.inf)
                            # if x2.get("F")[0] <= 1:
                            #    dists.append(fp) #F=1-fp  0.7
                            # else: #face not found #image 2 outside cluster
                            # dists.append(math.inf) #F=1-inf (-inf)
                            # else
                    closestDist = min(dists)
                    dists.remove(closestDist)
                    closestDist = min(dists)
                    fy_ = 1 - closestDist
                else:
                    # print("Comparing Current vs. Current, F=1-2nClosest")
                    # print("Previous Population not found..")
                    # if previous population is None, closestDist is based on current population
                    dists_2 = []
                    for x3 in X:
                        imgPath, F3, new_img, new_label = generateAnImage(x3, caseFile)
                        imgPath += ".png"
                        if F3:
                            N, DNNResult, P, L, D3, _ = doImage(imgPath, caseFile, CH, [new_img, new_label])
                            f3 = D3 / CR
                            fp = doParamDist(x, x3, xl, xu)
                            #d_sim.append(fp)
                            #d_hm.append(D3)
                            if f3 <= 1:
                                dists_2.append(fp)
                    closestDist = min(dists_2)
                    dists_2.remove(closestDist)
                    if len(dists_2) >= 1:
                        closestDist = min(dists_2)
                    fy_ = 1 - closestDist
            else:
                fy_ = f1
        f1_.append(f1)
        fy.append([fy_])
    # print("F1:", f1_)

    return fy


def HUDDevaluate_Pop2(X, caseFile, Y):
    print("POP2")
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        j = len(fy)
        l_D = []
        for j in range(0, len(Y)):
            l_D.append(doParamDist(x, Y[j].X, setX(1, "L"), setX(1, "U")))
        closestIndex = np.argsort(np.array(l_D))[0]
        fyy = []
        fy_ = math.inf
        if x is None:
            for x2 in Y:
                fyy.append(fy_)
            continue
        imgPath, F, new_img, new_label = generateAnImage(x, caseFile)
        imgPath += ".png"
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH, [new_img, new_label])
            f1 = D / CR

            if f1 <= 1:
                j = 0
                for x2 in Y:
                    imgPath2, F2, new_img2, new_label2 = generateAnImage(x2.X, caseFile)
                    imgPath2 += ".png"
                    N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH, [new_img2, new_label2])
                    _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer], [new_img, new_label])
                    fp = doParamDist(x, x2.X, xl, xu)
                    if j == closestIndex:
                        if not DNNResult:
                            fy_ = fp
                            fyy.append(fy_)
                            print("Eval-OK:", closestIndex, fy_)
                        else:
                            fy_ = 2 - N
                            fyy.append(fy_)
                            print("Eval-NO:", closestIndex, fy_)
                    else:
                        fy_ = 2 + fp
                        fyy.append(fy_)
                    j += 1
            else:
                for x2 in Y:
                    fy_ = 3 + f1
                    fyy.append(fy_)
        else:
            for x2 in Y:
                fyy.append(fy_)
        fy.append(fyy)
    return fy


def HUDDevaluate_Pop2_A(X, caseFile, Y):
    print("POP2")
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        j = len(fy)
        if x is None:
            fy.append(math.inf)
            continue
        imgPath, F, new_image, new_label = generateAnImage(x, caseFile)
        imgPath += ".png"
        fy_ = math.inf
        f1 = math.inf
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH, [new_image, new_label])
            f1 = D / CR
            if f1 <= 1:
                if not DNNResult:
                    x2 = Y[j]
                    imgPath2, F2, new_image2, new_label2 = generateAnImage(x2.X, caseFile)
                    imgPath2 += ".png"
                    N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH, [new_image2, new_label2])
                    _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer], [new_image, new_label])
                    fp = doParamDist(x, x2.X, xl, xu)
                    fy_ = fp
                else:
                    fy_ = 2 - N
            else:
                fy_ = 2 + f1
        f1_.append(f1)
        fy.append([fy_])
    print("F1:", f1_)
    return fy


def HUDDevaluate_Pop3(X, caseFile, Y):
    print("POP3")
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        j = len(fy)
        fy_ = math.inf
        fyy = []
        l_D = []
        for j in range(0, len(Y)):
            l_D.append(doParamDist(x, Y[j].X, setX(1, "L"), setX(1, "U")))
        closestIndex = np.argsort(np.array(l_D))[0]
        if x is None:
            for x2 in Y:
                fyy.append(fy_)
            continue
        imgPath, F, new_img, new_label = generateAnImage(x, caseFile)
        imgPath += ".png"
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH, [new_img, new_label])
            f1 = D / CR
            j = 0
            for x2 in Y:
                imgPath2, F2, new_img2, new_label2 = generateAnImage(x2.X, caseFile)
                imgPath2 += ".png"
                N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH, [new_img2, new_label2])
                _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer], [new_img, new_label])
                fp = doParamDist(x, x2.X, xl, xu)
                if j == closestIndex:
                    if DNNResult:
                        fy_ = fp
                        fyy.append(fy_)
                        print("Eval-OK:", closestIndex, fy_)
                    else:
                        fy_ = 1 + N + abs(1 - f1)
                        fyy.append(fy_)

                        print("Eval-NO:", closestIndex, fy_)
                else:
                    fy_ = 2 + fp
                    fyy.append(fy_)
                j += 1
        else:
            for x2 in Y:
                fyy.append(math.inf)
        fy.append(fyy)
    return fy


def HUDDevaluate_N(x, caseFile, prevEN, y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    #print(generateAnImage(x, caseFile))
    imgPath, F, new_img, new_label = generateAnImage(x, caseFile)
    imgPath += ".png"
    f2_ = None
    fy = math.inf
    fy_ = math.inf
    fp = None
    ND = None
    D2 = None
    if F:
        N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH, [new_img, new_label])
        print(N)
        f1 = D / CR
        fp = doParamDist(x, x, xl, xu)
        if DNNResult:
            AN = N
        else:
            AN = 1

        if f1 <= 1:
            f2 = 1 - AN
            f1_ = 1 - f1
        else:
            f2 = 1
            f1_ = f1

        if prevEN is not None:

            if not DNNResult:
                f2_ = 0
            else:
                f2_ = 1
        if y is not None:

            for y1 in y:
                imgPath2, F2, new_img, new_label = generateAnImage(y1.X, caseFile)
                imgPath2 += ".png"
                if F2:
                    N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH, [new_img, new_label])
                    _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer], [new_img, new_label])

                    ND = D2 / CR
                    fp = doParamDist(x, y1.X, xl, xu)
                    ND = fp
                    if math.isnan(fp):
                        ND = 0
                    if f1 < 1:
                        if not DNNResult:
                            fy = ND
                        else:
                            fy = 1 + (1 - N)
                    else:
                        fy = 2 + f1

                    if DNNResult:
                        fy_ = ND
                    else:
                        fy_ = 1 + N

                else:
                    print("face 2 not found")
    else:
        f1 = math.inf
        f1_ = math.inf
        f2 = math.inf
        f2_ = math.inf
        fy = math.inf
        fp = math.inf
        N = math.inf
        D = math.inf
        D2 = math.inf
        P = None
        L = None
        DNNResult = None
        AN = None
    problemDict = {"Face": F, "Prediction": P, "Label": L, "N_": N, "Adjusted_N": AN, "FY_": fy_,
                   "Medoid-Distance": D, "Y-Distance": D2, "DNNResult": DNNResult,
                   "F1": f1, "F1_": f1_, "F2": f2, "F2_": f2_, "FY": fy, "FP": fp, "ND": ND}
    #print(problemDict)
    return problemDict


class HUDDProblem(Problem):
    def __init__(self, caseFile, probID, prevEN, prevPOP, n_gen, **kwargs):
        self.outPath = caseFile["outputPath"]
        self.problemDict = {}
        self.n_gen = n_gen
        self.clusterRadius = caseFile["CR"]
        self.centroidHM = caseFile["CH"]
        self.caseFile = caseFile
        self.probID = probID
        self.prevEN = prevEN
        self.r = 0
        self.prevPOP = prevPOP
        self.t = time.time()
        self.counter = 0
        if probID == 1:
            n_var = nVar
            n_obj = 2
            xl = setX(1, "L")
            xu = setX(1, "U")
        else:
            if probID == 2:
                n_obj = 1  # FIXME
            else:
                n_obj = indvdSize
            n_var = indvdSize * nVar  # FIXME
            xl = setX(indvdSize, "L")  # FIXME
            xu = setX(indvdSize, "U")  # FIXME
        self.xl = xl
        self.xu = xu
        print("Solving Problem .. ", probID)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True, **kwargs,
                         type_var=np.float)

    def _evaluate(self, x, out, *args, **kwargs):
        self.counter += 1
        print(str(100 * (self.counter / self.n_gen))[0:5] + "%", end="\r")
        if not exists(self.outPath):
            makedirs(self.outPath)
        if self.probID == 1:
            problemDict = HUDDevaluate(x, self.caseFile, None, None)
            self.problemDict[processX(x)] = problemDict
            out["F"] = np.array([problemDict["F1"], problemDict["F2"]])  # FIXME
        else:
            problemDict = HUDDevaluate2(x, self.caseFile, self.prevEN, self.prevPOP[self.r])
            if self.r < len(self.prevPOP) - 1:
                self.r += 1
            else:
                self.r = 0
            if self.probID == 2:
                out["F"] = getF(problemDict, "F3")  # FIXME
            else:
                out["F"] = getF(problemDict, "FY")
        # if time.time() - self.t > 2h:
        # terminate


def evaluateResults(caseFile):
    cID = input("Enter RCC ID:")
    csvPath = join(caseFile["filesPath"], "GeneratedImages", cID, "results.csv")
    cFile = join(caseFile["filesPath"], "GeneratedImages", cID, "config.pt")
    if isfile(cFile):
        cPART = torch.load(cFile)
    else:
        cPART = {}
        x = int(input("Enter # of rules:.. "))
        portions = []
        rules = []
        val1x = []
        val2x = []
        if x == 0:
            portions.append([1.0])
            rules.append([1e9])
        for j in range(0, x):
            portion = float(input("Enter portion (x/y):.. "))
            n = input("Enter # of params:.. ")
            val1 = []
            val2 = []
            param = []
            for i in range(0, int(n)):
                param.append(input(
                    "Choose param: 1-3: cam_dir -- 4-6: cam_loc -- 7-9: lamp_loc -- 10-12: head_pose -- 13: face_model:.. "))
                val1.append(input("Enter Unsafe Value1:.. "))
                val2.append(input("Enter Unsafe Value2:.. "))
            val1x.append(val1)
            val2x.append(val2)
            rules.append(param)
            portions.append(portion)
        cPART['rules'] = rules
        cPART['portions'] = portions
        cPART['val1x'] = val1x
        cPART['val2x'] = val2x
        torch.save(cPART, cFile)
    imageList = pd.read_csv(csvPath)
    paramNameList = ["cam_dir0", "cam_dir1", "cam_dir2", "cam_loc0", "cam_loc1", "cam_loc2", "lamp_loc0", "lamp_loc1",
                     "lamp_loc2",

                     "head_pose0", "head_pose1",
                     "head_pose2", "pose"]
    paramNameListX = ["CamDir_X", "CamDir_Y", "CamDir_Z", "CamLoc_X", "CamLoc_Y", "CamLoc_Z", "LampLoc_X", "LampLoc_Y",
                      "LampLoc_Z",

                      "HeadPose_X", "HeadPose_Y",
                      "HeadPose_Z", "faceModel"]
    paramDict = {}
    for j in range(0, nVar):
        paramDict[paramNameList[j]] = []
    for index, row in imageList.iterrows():
        if not row["DNNResult"]:
            for i in range(0, nVar):
                paramDict[paramNameList[i]].append(float(row[paramNameList[i]]))
    paramFlag = False
    n = 0
    #print(paramNameList)
    for param in paramDict:
        for j in range(0, len(cPART['rules'])):
            for i in range(0, len(cPART['rules'][j])):

                #print(int(cPART['rules'][j][i]))
                if int(cPART['rules'][j][i]) < 1e9 and param == paramNameList[int(cPART['rules'][j][i]) - 1]:
                    print("R" + str(j) + ":", str(cPART['val1x'][j][i])[0:6], "<", paramNameListX[n], "<",
                          str(cPART['val2x'][j][i])[0:6])
                    if float(cPART['val1x'][j][i]) <= -1e8:
                        cPART['val1x'][j][i] = min(paramDict[param])
                    if float(cPART['val2x'][j][i]) >= 1e8:
                        cPART['val2x'][j][i] = max(paramDict[param])

                else:
                    paramFlag = True

        if paramFlag or len(cPART['rules'][0]) == 0:
            print(str(min(paramDict[param]))[0:6], "<", paramNameListX[n], "<", str(max(paramDict[param]))[0:6])
        n += 1

    # PR = input("Precision/Recall?: Y/N")
    eval_imgs = input("Enter # of evaluation images:.. ")
    # poseMode = input("Enter Evaluation mode (1: TrainingSet - 2: TestSet):")
    caseFile["SimDataPath"] = join(caseFile["filesPath"], "Pool")
    # csv2 = join(caseFile["filesPath"], "DT_MC_CC.csv")
    # csv3 = join(caseFile["filesPath"], "DT_RCC_CC.csv")
    # iL = pd.read_csv(csv2)
    # iL2 = pd.read_csv(csv3)
    pL = ["cam_look_direction_0", "cam_look_direction_1", "cam_look_direction_2", "cam_loc_0", "cam_loc_1", "cam_loc_2",
          "", "", "", "", "", "", "", "", "", "", "head_pose_0", "head_pose_1", "head_pose_2"]
    RM = 0
    RR2 = 0
    A = 0
    M = 0
    R2 = 0
    R = 0
    # if PR == "Y":
    #    for index, row in iL.iterrows():
    #        A += 1
    #        if (float(val1[0]) <= float(row[str(pL[int(param[0]) - 1])]) <= float(val2[0])) and \
    #                (float(val1[1]) <= float(row[str(pL[int(param[1]) - 1])]) <= float(val2[1])) and \
    #                (float(val1[2]) <= float(row[str(pL[int(param[2]) - 1])]) <= float(val2[2])):
    #            R += 1
    #        if int(row["clusterID"]) != 0:
    #            M += 1

    #            if (float(val1[0]) <= float(row[str(pL[int(param[0]) - 1])]) <= float(val2[0])) and \
    #                    (float(val1[1]) <= float(row[str(pL[int(param[1]) - 1])]) <= float(val2[1])) and \
    #                    (float(val1[2]) <= float(row[str(pL[int(param[2]) - 1])]) <= float(val2[2])):
    #                RM += 1
    #    for index, row in iL2.iterrows():
    #        if int(row["clusterID"]) == int(cID):
    #            R2 += 1
    #            if (float(val1[0]) <= float(row[str(pL[int(param[0]) - 1])]) <= float(val2[0])) and \
    #                    (float(val1[1]) <= float(row[str(pL[int(param[1]) - 1])]) <= float(val2[1])) and \
    #                    (float(val1[2]) <= float(row[str(pL[int(param[2]) - 1])]) <= float(val2[2])):
    #                RR2 += 1
    #    print(A, M,  R, RM, R2, RR2)
    #    print("MC_Precision:", 100* RM/R)
    #    print("MC_Recall:", 100 * RM/M)
    #    print("RCC_Precision:", 100* RR2/R)
    #    print("RCC_Recall:", 100 * RR2/R2)
    outDir = join(caseFile["filesPath"], "Evaluation", cID)
    if not exists(outDir):
        makedirs(outDir)
    # n2 = 0
    f_ = 0
    clusterImages = []
    n = 0
    totalimgs = 0
    DNNResult = None
    F = None
    t = time.time()
    for j in range(0, len(cPART['rules'])):
        toEval = int(cPART['portions'][j] * int(eval_imgs))
        print("portion:", cPART['portions'][j])
        print("Generating", toEval, " images")
        total = 0
        while total < toEval:
            x = setX(1, "R")
            print(total, DNNResult, F, str(time.time() - t)[0:5], end="\r")
            t = time.time()
            #x = setNewX(x, paramDict, paramNameList, cPART['rules'][j], cPART['val1x'][j], cPART['val2x'][j])
            # if int(poseMode) == 2:
            #    x[19] = random.randint(0, 8)
            # elif int(poseMode) == 1:
            #    x[19] = random.randint(0, 2)
            #break

            imgPath, F, new_img, new_label = generateAnImage(x, caseFile)
            #print("eval, new_img", new_img)
            if not F:
                f_ += 1
                # toEval += 1
            else:
                imgPath += ".png"
                N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, None, [new_img, new_label])
                # DNNResult2, pred = testModelForImg(caseFile["DNN2"], L, imgPath, caseFile)
                if DNNResult:
                    n += 1
                # if DNNResult2:
                #    n2 += 1
                total += 1
                totalimgs += 1
                clusterImages.append(imageio.imread(imgPath))
    imageio.mimsave(join(outDir, "SEDE_" + str(len(clusterImages)) + "_" + str(100 * (n / totalimgs))[0:5] + '.gif'),
                    clusterImages)
    print("Accuracy:", 100 * (n / (totalimgs)))
    # print("Accuracy:", 100* (n2/(total)))
    print("not found", f_)
    conversions = np.array([1914, int(n)])
    clicks = np.array([2200, int(totalimgs)])
    zscore, pvalue = proportions_ztest(conversions, clicks, alternative='two-sided')
    print('zscore = {:.4f}, pvalue = {:.4f}'.format(zscore, pvalue))

    ob_table = np.array([[2200, 1914], [int(totalimgs), int(n)]])
    result = stats.chi2_contingency(ob_table, correction=False)  # correction = False due to df=1
    chisq, pvalue = result[:2]
    print('chisq = {}, pvalue = {}'.format(chisq, pvalue))


def setNewX(x, paramDict, paramNameList, param, val1, val2):
    paramNameListX = ["CamDir_X", "CamDir_Y", "CamDir_Z", "CamLoc_X", "CamLoc_Y", "CamLoc_Z", "LampLoc_X", "LampLoc_Y",
                      "LampLoc_Z",

                      "HeadPose_X", "HeadPose_Y",
                      "HeadPose_Z", "faceModel"]
    for j in range(0, nVar):
        minVal = min(paramDict[paramNameList[j]])
        maxVal = max(paramDict[paramNameList[j]])
        valMin = float(minVal)
        valMax = float(maxVal)
        if j == (nVar-1):
            maxVal += 0.99  # we round down the facemodel value
        x[j] = random.uniform(minVal, maxVal)
        for z in range(0, len(param)):
            if j == (int(param[z]) - 1):
                if j == nVar-1:
                    val2[z] = float(val2[z]) + 0.99
                if float(val1[z]) < minVal:
                    val1[z] = minVal
                if float(val2[z]) > maxVal:
                    val2[z] = maxVal
                # print(val1[z], val2[z])
                if float(val1[z]) > float(val2[z]):
                    "error in parameters settings"
                x[j] = random.uniform(float(val1[z]), float(val2[z]))
                valMin = float(val1[z])
                valMax = float(val2[z])
        if valMin < 0 and valMax <0:
            print(str(valMax)[0:6], ">=" ,paramNameListX[j], ">=" ,str(valMin)[0:6])
        else:
            print(str(valMin)[0:6], ">=" ,paramNameListX[j], ">=" ,str(valMax)[0:6])
    return x


def compareX(x1, x2):
    equal = True
    for i in range(0, len(x1)):
        if x1[i] != x2[i]:
            equal = False
    return equal


def HUDDevaluate(x, caseFile, prevEN, y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    imgPath, F, new_img, new_label = generateAnImage(x, caseFile)
    imgPath += ".png"
    f2_ = None
    fy = math.inf
    fp = None
    ND = None
    D2 = None
    HN = None
    LN = False
    if F:
        N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH, [new_img, new_label])

        f1 = D / CR

        if DNNResult:
            AN = N
        else:
            AN = 1

        if f1 <= 1:
            f2 = 1 - AN
            f1_ = 1 - f1
        else:
            f2 = 1
            f1_ = f1

        if prevEN is not None:
            if AN >= max(prevEN):
                HN = True
            else:
                HN = False

            if (not DNNResult) or HN:
                f2_ = 0
            else:
                f2_ = 1

        if y is not None:
            imgPath2, F2, new_img, new_label = generateAnImage(y, caseFile)
            imgPath2 += ".png"
            if F2:
                N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH, [new_img, new_label])
                N1, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer], [new_img, new_label])

                if N1 <= N2:
                    LN = True
                    HN = False
                ND = D2 / CR
                fp = doParamDist(x, y, xl, xu)
                ND = fp
                if DNNResult and LN:
                    fy = ND
                elif (not DNNResult) or (not LN):
                    if ND != 0:
                        # fy = 1 + (1 / f1)
                        fy = 1 + (N / f1)
                    else:
                        fy = math.inf
            else:
                print("face 2 not found")
    else:
        f1 = math.inf
        f1_ = math.inf
        f2 = math.inf
        f2_ = math.inf
        fy = math.inf
        fp = math.inf
        N = math.inf
        D = math.inf
        D2 = math.inf
        LN = False
        HN = True
        P = None
        L = None
        DNNResult = None
        AN = None
    problemDict = {"Face": F, "Prediction": P, "Label": L, "N_": N, "Adjusted_N": AN, "Low_N": LN, "High_N": HN,
                   "Medoid-Distance": D, "Y-Distance": D2, "DNNResult": DNNResult,
                   "F1": f1, "F1_": f1_, "F2": f2, "F2_": f2_, "FY_": fy, "FP": fp, "ND": ND}
    return problemDict


def HUDDevaluate2(x, caseFile, prevEN, prevPOP):
    global indvdSize
    f1l = []
    f2_l = []
    fyl = []
    indiv = []
    N_ = []
    PS = 0
    problemDict = {processX(x): {}}
    for i in range(0, indvdSize):
        x1 = getI(x, i)
        indiv.append(x1)
        X1 = processX(x1)
        Y1 = None
        if prevPOP is not None:
            Y1 = getI(prevPOP.X, i)
        problemDict2 = HUDDevaluate(x1, caseFile, prevEN, Y1)
        problemDict[X1] = problemDict2
        if problemDict2["Face"]:
            N_.append(problemDict2["Adjusted_N"])
            f1l.append(problemDict2["F1"])
            f2_l.append(problemDict2["F2_"])
            if problemDict2["DNNResult"] and problemDict2["Low_N"]:
                PS += 1
        fyl.append(problemDict2["FY_"])
    ANPD = getANPD(indiv, caseFile, True)
    MP = f2_l.count(0) / len(f2_l)
    if PS == 0:
        PS = 1
    else:
        PS = 1
    IP = 0
    for z in f1l:
        if z <= 1.0:
            IP += 1
    IP = IP / len(f1l)
    if prevEN is not None and (len(f1l) == indvdSize):
        if float(max(f1l)) <= 1.0 and float(max(f2_l)) == 0.0:
            f3 = [1 - ANPD]
        else:
            if ANPD == 0 or IP == 0 or MP == 0:
                f3 = [math.inf]
            else:
                f3 = [1 + ((1 / (IP)) * (1 / MP) * math.log10(1 / ANPD))]
    else:
        f3 = [math.inf]
    FY = [i * PS for i in fyl]
    for any_ in problemDict:
        problemDict[any_]["F3"] = f3
        problemDict[any_]["FY"] = FY
        problemDict[any_]["ANPD"] = ANPD

    print("F1:", max(f1l))
    print("FY:", FY)
    print("%:", str(100 * MP)[0:5] + "%")
    print("F3:", f3)
    print("IP:", str(100 * IP)[0:5] + "%")
    problemDict[processX(x)]["N"] = N_
    # print("Individual -- F3 = ", f3)
    # print("Individual -- Fy = ", fyl)
    return problemDict


def prepareINIT(PF, caseFile, dirPath, probNum, pop_size):
    print("Preparing Initial Population", probNum)
    CP = join(caseFile["GI"], str(caseFile["ID"]), "init" + str(probNum) + ".pop")
    N, _ = getEntropy(PF, probNum, caseFile)
    if isfile(CP):
        initpop = loadCP(CP)
        print("Loaded Initpop:", initpop)
    else:
        initpop = initPOP_Pop(PF, probNum, pop_size, N, caseFile)
        saveCP(initpop, CP)

    exportResults(initpop, "/Initial/" + str(probNum) + ".txt", "/Initial/" + str(probNum), probNum, dirPath,
                  None, caseFile, N, initpop, False)
    print("Initial population Size:", len(initpop))
    return initpop, N


def getEntropy(PF, probNum, caseFile):
    N = []
    PF2 = []
    for member2 in PF:
        problemDict = HUDDevaluate_N(member2.X, caseFile, None, None)
        if problemDict["N_"] is not None:
            N.append(problemDict["N_"])
    return N, PF2


def initPOP(PF, probNum, popLen, N, caseFile):
    pop2 = Population(popLen)
    j = 0
    for i in range(0, popLen):
        if probNum == 2:
            newX = concatX(PF)
            key = "F3"
        else:
            newX = PF[j].X
            key = "FY"
        cleanedX = cleanX(newX, caseFile)
        problemDict = HUDDevaluate2(cleanedX, caseFile, N, None)
        pop2[i] = Individual(X=np.array(cleanedX), CV=PF[j].CV, feasible=PF[j].feasible, F=getF(problemDict, key))
        if j < len(PF) - 1:
            j += 1
        else:
            j = 0
    return pop2


def initPOP_N(PF, probNum, popLen, N, caseFile):
    pop2 = Population(popLen)
    for i in range(0, len(PF)):
        F_list = []
        if probNum == 2:
            key = "FY"
        elif probNum == 3:
            key = "FY_"
        for j in range(0, len(PF)):
            problemDict = HUDDevaluate_N(PF[i].X, caseFile, N, PF[j].X)
            F_list.append(problemDict[key])
        pop2[i] = Individual(X=PF[i].X, CV=PF[i].CV, feasible=PF[i].feasible, F=np.array(F_list))
    return pop2


def initPOP_Pop(PF, probNum, popLen, N, caseFile):
    pop2 = Population(popLen)
    X_list = []
    for ind in PF:
        X_list.append(ind.X)
    if probNum == 1:
        F = HUDDevaluate_Pop1(X_list, caseFile, PF)
    elif probNum == 2:
        F = HUDDevaluate_Pop2(X_list, caseFile, PF)
        print("F", F)
    else:
        F = HUDDevaluate_Pop3(X_list, caseFile, PF)
    for i in range(0, len(PF)):
        pop2[i] = Individual(X=PF[i].X, CV=PF[i].CV, feasible=PF[i].feasible, F=np.array(F[i]))
    return pop2


def getPF(pop1, CP):
    if isfile(CP):
        PF = loadCP(CP)
    else:
        print("Extracting Pareto Front")
        length = 0
        PF_ = []
        for i in range(0, len(pop1)):
            if pop1[i].get("rank") == 0:
                PF_.append(pop1[i])
                length += 1
        PF = Population(len(PF_))
        for i in range(0, len(PF)):
            PF[i] = PF_[i]
        print("Pareto Front #individuals:", length)
        saveCP(PF, CP)
    return PF


def cleanX(x, caseFile):
    goodIndex = []
    newX = []
    for j in range(0, indvdSize):
        _, face, _ = generateAnImage(getI(x, j), caseFile)
        if face:
            goodIndex.append(j)
    for j in range(0, indvdSize):
        if j in goodIndex:
            for z in getI(x, j):
                newX.append(z)
        else:
            newGoodIndex = goodIndex[random.randint(0, len(goodIndex) - 1)]
            goodIndex.append(newGoodIndex)
            for z in getI(x, newGoodIndex):
                newX.append(z)
    return newX


def exportResults_2(PF, txt1, outPathPOP, popNum, dirPath, outFile, caseFile, prev_en, prev_pop, csvFlag):
    outPathPOP = dirPath + outPathPOP
    if not exists(outPathPOP):
        makedirs(outPathPOP)
    SimDataPath = caseFile["SimDataPath"]
    clusterImages = list()
    txt1 = dirPath + txt1
    file = open(txt1, "a")
    # if popNum == 1:
    #    F = HUDDevaluate_Pop1(PF.get("X"), caseFile, None)
    # if popNum == 2:
    #    F = HUDDevaluate_Pop2(PF.get("X"), caseFile, PF.get("X"))
    # if popNum == 3:
    #    F = HUDDevaluate_Pop3(PF.get("X"), caseFile, PF.get("X"))

    dub = 0
    for i in range(0, len(PF)):
        X = PF[i].X
        problemDict = HUDDevaluate(X, caseFile, None, None)
        if problemDict["Face"]:
            imgName = processX(X) + ".png"
            if not isfile(join(outPathPOP, imgName)):
                shutil.copy(join(SimDataPath, imgName), join(outPathPOP, imgName))
            else:
                shutil.copy(join(SimDataPath, imgName), join(outPathPOP, processX(X) + "_" + str(dub) + ".png"))
                dub += 1
            if popNum == 0 or popNum == 1:
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
            if popNum == 2 and not problemDict["DNNResult"]:
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
            if popNum == 3 and problemDict["DNNResult"]:
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
            file.write(imgName + "\n")
            file.write(str(PF[i].F) + "\n")
            for nameX in problemDict:
                file.write(str(nameX) + ": " + str(problemDict[nameX]) + "\n")
    if csvFlag:
        toCSV_N(PF, outFile, caseFile, prev_en, prev_pop, popNum)
    if len(clusterImages) > 1:
        imageio.mimsave(outPathPOP + "_" + str(int(len(clusterImages)/3)) + '.gif', clusterImages)


def exportResults(PF, txt1, outPathPOP, popNum, dirPath, outFile, caseFile, prev_en, prev_pop, csvFlag):
    outPathPOP = dirPath + outPathPOP
    if not exists(outPathPOP):
        makedirs(outPathPOP)
    SimDataPath = caseFile["SimDataPath"]
    clusterImages = list()
    txt1 = dirPath + txt1
    file = open(txt1, "a")
    if popNum == 0 or popNum == 1 or popNum == 2 or popNum == 3:
        dub = 0
        for i in range(0, len(PF)):
            X = PF[i].X

            #print(X)
            problemDict = HUDDevaluate(X, caseFile, None, None)
            if problemDict["Face"]:
                imgName = processX(X) + ".png"
                if not isfile(join(outPathPOP, imgName)):
                    shutil.copy(join(SimDataPath, imgName), join(outPathPOP, imgName))
                else:
                    shutil.copy(join(SimDataPath, imgName), join(outPathPOP, processX(X) + "_" + str(dub) + ".png"))
                    dub += 1
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                file.write(imgName + "\n")
                file.write(str(PF[i].F) + "\n")
                for nameX in problemDict:
                    file.write(str(nameX) + ": " + str(problemDict[nameX]) + "\n")
        if csvFlag:
            toCSV_N(PF, outFile, caseFile, prev_en, prev_pop, popNum)
        if len(clusterImages) > 1:
            imageio.mimsave(outPathPOP + "_" + str(len(clusterImages)) + '.gif', clusterImages)
    else:
        dub = 0
        for i in range(0, len(PF)):
            clusterImages = list()
            if prev_pop is not None:
                problemDict = HUDDevaluate2(PF[i].X, caseFile, prev_en, prev_pop[i])
            else:
                problemDict = HUDDevaluate2(PF[i].X, caseFile, prev_en, None)
            for j in range(0, indvdSize):
                X = getI(PF[i].X, j)
                if problemDict[processX(X)]["Face"]:
                    imgName = processX(X) + ".png"
                    clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                    if not isfile(join(outPathPOP, imgName)):
                        shutil.copy(join(SimDataPath, imgName), join(outPathPOP, imgName))
                    else:
                        shutil.copy(join(SimDataPath, imgName), join(outPathPOP, processX(X) + "_" + str(dub) + ".png"))
                        dub += 1
                    file.write(imgName + "\n")
                    file.write(str(PF[i].F) + "\n")
                    for nameX in problemDict[processX(X)]:
                        file.write(str(nameX) + ": " + str(problemDict[processX(X)][nameX]) + "\n")
            if len(clusterImages) > 1:
                imageio.mimsave(outPathPOP + str(i) + "_" + str(len(clusterImages)) + '.gif', clusterImages)
        if csvFlag:
            toCSV_N(PF[0], outFile, caseFile, prev_en, prev_pop[0], popNum)
    file.close()


def toCSV_N(pop, outFile, CF, prevEN, prev_pop, probNum):
    ID = CF["ID"]
    counter = 1
    problemDict = HUDDevaluate_N(pop[0].X, CF, prevEN, prev_pop)
    if probNum == 2:
        strW = "idx,clusterID"
        for _x_ in problemDict:
            if "Face" in problemDict:
                if problemDict["Face"]:
                    for nameX in problemDict:
                        if nameX != "FY":
                            strW += "," + str(nameX)
                    break
        strW += ",cam_dir0,cam_dir1,cam_dir2,cam_loc0,cam_loc1,cam_loc2,lamp_loc0," \
                "lamp_loc1,lamp_loc2," \
                "head_pose0,head_pose1,head_pose2,pose,imgPath\r\n"
        outFile.writelines(strW)
    if probNum == 3:
        ID = 0
    for j in range(0, len(pop)):

        problemDict = HUDDevaluate_N(pop[j].X, CF, prevEN, prev_pop)
        X = pop[j].X
        if problemDict["Face"]:
            if ((not problemDict["DNNResult"]) and (probNum == 2)) or ((problemDict["DNNResult"]) and (probNum == 3)):
                imgPath = join(CF["filesPath"], "Pool", processX(X) + ".png")
                strMerge = str(counter) + "," + str(ID)
                for nameX in problemDict:
                    if nameX != "FY":
                        strMerge += "," + str(problemDict[nameX])
                for j in range(0, len(X) - 1):
                    strMerge += "," + str(X[j])
                strMerge += "," + str(math.floor(X[len(X)-1]))
                strMerge += "," + str(imgPath)
                strMerge += "\r\n"
                outFile.writelines(strMerge)
            counter += 1


def toCSV(member, outFile, CF, prevEN, prev_pop, probNum):
    ID = CF["ID"]
    counter = 1
    problemDict = HUDDevaluate(member.X, CF, prevEN, prev_pop)
    if probNum == 2:
        strW = "idx,clusterID"
        for _x_ in problemDict:
            if "Face" in problemDict[_x_]:
                if problemDict[_x_]["Face"]:
                    for nameX in problemDict[_x_]:
                        if nameX != "FY":
                            strW += "," + str(nameX)
                    break
        strW += ",cam_dir0,cam_dir1,cam_dir2,cam_loc0,cam_loc1,cam_loc2,lamp_loc0," \
                "lamp_loc1,lamp_loc2," \
                "head_pose0,head_pose1,head_pose2,pose,imgPath\r\n"
        outFile.writelines(strW)
    if probNum == 3:
        ID = 0
    for j in range(0, indvdSize):
        X = getI(member.X, j)
        X_ = processX(X)
        if problemDict[X_]["Face"]:
            if probNum == 2 or ((problemDict[X_]["DNNResult"]) and (probNum == 3)):
                imgPath = join(CF["filesPath"], "Pool", processX(X) + ".png")
                strMerge = str(counter) + "," + str(ID)
                for nameX in problemDict[X_]:
                    if nameX != "FY":
                        strMerge += "," + str(problemDict[X_][nameX])
                for j in range(0, len(X) - 1):
                    strMerge += "," + str(X[j])
                strMerge += "," + str(math.floor(X[len(X)-1]))
                strMerge += "," + str(imgPath)
                strMerge += "\r\n"
                outFile.writelines(strMerge)
            counter += 1


def getANPD(pop, CF, paramFlag):
    xl = CF["xl"]
    xu = CF["xu"]
    CR = CF["CR"]
    global layer
    HMList = list()
    goodX = list()
    for i in range(0, len(pop)):
        imgPath, faceFound, _ = generateAnImage(pop[i], CF)
        imgPath = imgPath + ".png"
        if faceFound:
            layersHMX, entropy = generateHeatMap(imgPath, CF["DNN"], CF["datasetName"],
                                                 CF["outputPath"],
                                                 False, None, None, CF["imgExt"], None)
            HMList.append(layersHMX[layer])
            goodX.append(pop[i])
    numPairs = int((len(HMList) * (len(HMList) - 1)) / 2)
    D = np.zeros(numPairs)
    k = 0
    if paramFlag:
        for i in range(0, len(goodX)):
            for j in range(0, len(goodX)):
                if j > i:
                    D[k] = doParamDist(goodX[i], goodX[j], xl, xu)
                    k += 1
    else:
        for i in range(0, len(HMList)):
            for j in range(0, len(HMList)):
                if j > i:
                    D[k] = doDistance(HMList[i], HMList[j], "Euc") / (2 * CR)
                    k = k + 1
    ANPD = np.average(D)
    if ANPD == 0:
        return -math.inf
    if math.isnan(ANPD):
        return math.inf
    else:
        return ANPD


def getF(problemDict, key):
    for any_ in problemDict:
        return np.array(problemDict[any_][key])


def concatX(PF):
    z = []
    j = random.randint(0, len(PF) - 1)
    while len(z) < nVar * indvdSize:
        b = 0
        while b < 3:
            for x1 in PF[j].X:
                z.append(x1)
            if len(z) == nVar * indvdSize:
                break
            b += 1
        if j < len(PF) - 1:
            j += 1
        else:
            j = 0
    assert len(z) == nVar * indvdSize
    return z


def processX(x):
    out = str(x[0])[0:5]
    for i in range(1, len(x)):
        out += "_" + str(x[i])[0:5]
    return out


def setX(size, ID):
    _, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, headL, headU, faceL, faceU = getParamVals()
    xl = []
    if ID == "L":
        for i in range(0, size):
            for c in cam_dirL:
                xl.append(c)
            for c in cam_locL:
                xl.append(c)
            for c in lamp_locL:
                xl.append(c)
            for c in headL:
                xl.append(c)
            xl.append(faceL)
    elif ID == "U":
        xl = []
        for i in range(0, size):
            for c in cam_dirU:
                xl.append(c)
            for c in cam_locU:
                xl.append(c)
            for c in lamp_locU:
                xl.append(c)
            for c in headU:
                xl.append(c)
            xl.append(faceU)
    elif ID == "R":
        xl = []
        for i in range(0, size):
            for z in range(0, len(cam_dirU)):
                xl.append(random.uniform(cam_dirL[z], cam_dirU[z]))
            for z in range(0, len(cam_locU)):
                xl.append(random.uniform(cam_locL[z], cam_locU[z]))
            for z in range(0, len(lamp_locU)):
                xl.append(random.uniform(lamp_locL[z], lamp_locU[z]))
            for z in range(0, len(headU)):
                xl.append(random.uniform(headL[z], headU[z]))
            xl.append(random.uniform(faceL, faceU))
    return np.array(xl)


def getI(x, i):
    return x[nVar * i:nVar * (i + 1)]


def doParamDist(x, y, xl, xu):
    x_ = []
    y_ = []
    for m in range(0, len(x)):
        if xu[m] == xl[m]:
            x_.append(0)
            y_.append(0)
        else:
            x_.append((x[m] - xl[m]) / (xu[m] - xl[m]))
            y_.append((y[m] - xl[m]) / (xu[m] - xl[m]))
    d = distance.cosine(x_, y_)
    if math.isnan(d):
        return 0
    return d


def getParams(mini, maxi, param, BL):
    param_1st = param[0]
    param_3rd = param[1]
    if param_1st < mini:
        param_1st = mini
    if param_1st > maxi:
        param_1st = maxi
    if param_3rd < mini:
        param_3rd = mini
    if param_3rd > maxi:
        param_3rd = maxi
    if BL:
        return random.uniform(mini, maxi), random.uniform(mini, maxi)
    else:
        return param_1st, param_3rd


def getParamVals():
    param_list = ["cam_look_0", "cam_look_1", "cam_look_2", "cam_loc_0", "cam_loc_1", "cam_loc_2",
                  "lamp_loc_0", "lamp_loc_1", "lamp_loc_2", "head_0", "head_1", "head_2", "face"]
    # TrainingSet parameters (min - max)
    cam_dirL = [-0.10, -4.67, -1.69]
    # cam_dirL = [-0.08, -4.29, -1.27] # constant
    cam_dirU = [-0.08, -4.29, -1.27]
    cam_locL = [0.261, -5.351, 14.445]
    # cam_locL = [0.293, -5.00, 14.869] # constant
    cam_locU = [0.293, -5.00, 14.869]  # constant
    lamp_locL = [0.361, -5.729, 16.54]
    # lamp_locL = [0.381, -5.619, 16.64] # constant
    lamp_locU = [0.381, -5.619, 16.64]  # constant
    lamp_colL = [1.0, 1.0, 1.0]  # constant
    lamp_colU = [1.0, 1.0, 1.0]  # constant
    lamp_dirL = [0.873, -0.87, 0.698]  # constant
    lamp_dirU = [0.873, -0.87, 0.698]  # constant
    lamp_engL = 1.0  # constant
    lamp_engU = 1.0  # constant
    headL = [-41.86, -79.86, -64.30]
    headU = [36.87, 75.13, 51.77]
    faceL = 0
    #faceL = 0
    faceU = 5
    # TestSet parameters (min - max)
    # headL = [-32.94, -88.10, -28.53]
    # headU = [33.50, 74.17, 46.17]
    # fixing HP_2
    # headL = [-32.94, -88.10, -0.000001]
    # headU = [33.50, 74.17, 0]
    return param_list, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, headL, headU, faceL, faceU


def getPosePath():
    model_folder = "mhx2"
    label_folder = "newlabel3d"
    pose_folder = "pose"
    model_file = ["Aac01_o", "Aaj01_o", "Aai01_c", "Aah01_o", "Aaf01_o", "Aag01_o", "Aab01_o", "Aaa01_o",
                  "Aad01_o"]  # TrainingSet
    # model_file = [model_folder + "/Aae01_o", model_folder + "/Aaa01_o"] #ImprovementSet1
    # model_file = [model_folder + "/Aae01_o"] #ImprovementSet2
    # model_file = ["Aad01_o"]  # TestSet
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aae01_o"] #TestSet1
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aah01_o"] #TestSet2
    label_file = ["aac01_o", "aaj01_o", "aai01_c", "aah01_o", "aaf01_o", "aag01_o", "aab01_o", "aaa01_o",
                  "aad01_o"]  # TrainingSet
    # label_file = [label_folder + "/aae01_o", label_folder + "/aaa01_o"] #ImprovementSet1
    # label_file = [label_folder + "/aae01_o"] #ImprovementSet2
    # label_file = ["aad01_o"]  # TestSet
    # label_file = [label_folder + "/aad01_o", label_folder + "/aae01_o"] #TestSet1
    # label_file = [label_folder + "/aad01_o", label_folder + "/aah01_o"] #TestSet2
    pose_file = ["Aga", "Pbo", "Mbg", "Ldh", "Gle", "Hcl", "Acr", "Acq", "Dho"]  # TrainingSet
    # pose_file = [pose_folder + "/Fga", pose_folder + "/Acq"] #ImprovementSet1
    # pose_file = [pose_folder + "/Fga"] #ImprovementSet2
    # pose_file = ["Dho"]  # TestSet
    # pose_file = [pose_folder + "/Dho", pose_folder + "/Fga"] #TestSet1
    # pose_file = [pose_folder + "/Dho", pose_folder + "/Ldh"] #TestSet2
    return model_folder, label_folder, pose_folder, model_file, label_file, pose_file


from torchvision.transforms import ToTensor
def generateAnImage(x, caseFile):
    global counter
    SimDataPath = caseFile["SimDataPath"]
    outPath = caseFile["outputPath"]
    model_folder, label_folder, pose_folder, model_file, label_file, pose_file = getPosePath()
    m = random.randint(0, len(pose_file) - 1)
    m = int(math.floor(x[len(x)-1]))
    print(pose_file[m])
    print(label_file[m])
    print(model_file[m])
    imgPath = join(SimDataPath, processX(x))
    filePath = DIR + "ieekeypoints.py"
    # filePath = "./ieekeypoints.py"
    t1 = time.time()
    if not isfile(imgPath + ".png"):
        # ls = subprocess.run(['ls', '-a'], capture_output=True, text=True).stdout.strip("\n")
        ls = subprocess.run(
            [str(blenderPath), "--background", "--verbose", str(0), "--python", str(filePath), "--", "--path",
             str(path), "--model_folder",
             str(model_folder), "--label_folder", str(label_folder), "--pose_folder", str(pose_folder), "--pose_file",
             str(pose_file[m]), "--label_file", str(label_file[m]), "--model_file", str(model_file[m]), "--imgPath",
             # str(imgPath)], stdout=subprocess.PIPE)
             str(imgPath)], capture_output=True, text=True).stderr.strip("\n")
        #print(ls)
        # print(process.stdout)
        # print(process.stderr)
    # process.wait()
    t2 = time.time()
    # print("Image Generation", str(t2-t1)[0:5])
    # print(imgPath)
    img = cv2.imread(imgPath + ".png")
    new_img = None
    new_label = None
    if img is None:
        print("image not found, not processed")
        return imgPath, False, None, None
    if len(img) > 128:
        faceFound, new_img, new_label = processImage(imgPath + ".png", join(outPath, "IEEPackage/clsdata/mmod_human_face_detector.dat"))

        configData = {"data": new_img, "label": new_label}
        torch.save(configData, imgPath + ".pt")
    else:
        configData = torch.load(imgPath + ".pt")
        new_img = configData["data"]
        new_label = configData["label"]
        faceFound = True
    if caseFile["datasetName"] == "FLD":
        if new_img is not None:
            #if not isfile(imgPath + ".pt"):
            #else:
            data_transform = transforms.Compose([ToTensor()])
            inputs = data_transform(new_img).unsqueeze(0)
            labels = new_label
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            #model = ieeRegister(model)
            model = caseFile["DNN"]
            predict = model(inputs)
            predict_cpu = predict.cpu()
            predict_cpu = predict_cpu.detach().numpy()
            predict_xy = DS._target(predict_cpu)

            inputs_cpu = inputs.cpu()
            inputs_cpu = inputs_cpu.detach().numpy()
            num_sample = 1

            diff = np.square(new_label - predict_xy)
            sum_diff = np.sqrt(diff[:, :, 0] + diff[:, :, 1])
            def update(img, x_p, y_p, x_t=0, y_t=0):
                height, width = img.shape[0], img.shape[1]
                for idx in [-1, 0, 1]:
                    px = max(min(x_p + idx, width - 1), 0)
                    if x_t > 0 and y_t > 0:
                        tx = max(min(x_t + idx, width - 1), 0)
                    for jdx in [-1, 0, 1]:
                        py = max(min(y_p + jdx, height - 1), 0)
                        if x_t > 0 and y_t > 0:
                            ty = max(min(y_t + jdx, height - 1), 0)

                        if width > py > 0 and height > px > 0:
                            img[py, px, 0] = 255
                            img[py, px, 1] = 155
                            img[py, px, 2] = 0

                        if x_t > 0 and y_t > 0:

                            if width > ty > 0 and height > tx > 0:
                                img[ty, tx, 0] = 0
                                img[ty, tx, 1] = 0
                                img[ty, tx, 2] = 255
                        #else:
                        #    print("unlabelled image")
                return img

            wlabel = []
            # inputs_cpu = inputs.cpu()
            # inputs_cpu = inputs_cpu.detach().numpy()
            # num_sample = inputs_cpu.shape[0]
            # img = inputs_cpu[0] * 255.
            # img = img[0, :]
            # img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            # max_error = np.max(sum_diff[0])
            avg_error = np.sum(sum_diff[0]) / len(sum_diff[0])
            worst_KP = 0
            label = 0
            worst_label = 0
            kps = []
            for KP in sum_diff[0]:
                kps.append(label)
                if KP > worst_KP:
                    worst_KP = KP
                    worst_label = label
                label += 1
            wlabel.append(worst_label)
            kps = wlabel
            #print(kps)
            #print(wlabel)
            i = 0
            for idx in range(num_sample):
                img = inputs_cpu[idx] * 255.
                img = img[0, :]
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                xy = predict_xy[idx]
                lab_xy = labels
                #print(lab_xy)
                for kp in kps:
                    #print(kp)
                    x_p = int(xy[kp, 0] + 0.5)
                    y_p = int(xy[kp, 1] + 0.5)
                    x_t = int(lab_xy[kp, 0] + 0.5)
                    y_t = int(lab_xy[kp, 1] + 0.5)
                    if x_t == 0 and y_t == 0:
                        faceFound = False
                        new_img = None
                        new_label = None
                    img = update(img, x_p, y_p, x_t, y_t)
                cv2.imwrite(imgPath+".png", img)
    # generator = ieeKP.IEEKPgenerator(model_folder, pose_folder, label_folder)
    # imgPath = generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
    #                                                   cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
    #                                                   label_file[m])
    # print("Image processing", str(time.time()-t2)[0:5])

    shutil.copy(imgPath.split(".png")[0] + ".pt", os.path.join(os.path.dirname(imgPath), "S" + str(counter-1) + ".pt"))
    return imgPath, faceFound, new_img, new_label


def doImage(imgPath, caseFile, centroidHM, new_img):
    #print(imgPath)
    layersHM, entropy_HPD = generateHeatMap(imgPath, caseFile["DNN"], caseFile["datasetName"], caseFile["outputPath"],
                                        False, None, caseFile["testDataNpy"], caseFile["imgExt"], None)
    lable = labelImage(imgPath)
    #print("doImage, new_img", new_img)
    if caseFile["datasetName"] == "FLD":

        DNNResult, layersHM, entropy_FLD = testModelForImg(caseFile["DNN"], lable, imgPath, caseFile, new_img)
    elif caseFile["datasetName"] == "HPD":

        DNNResult, _, _ = testModelForImg(caseFile["DNN"], lable, imgPath, caseFile, new_img)
    # if imgPath in Dist_Dict:
    #print("DNNResult", DNNResult)
    if centroidHM is None:
        dist = 0
    else:
        if layersHM is None:
            dist = math.inf
        else:
            dist = doDistance(centroidHM, layersHM[int(caseFile["selectedLayer"].replace("Layer", ""))], "Euc")
    # Dist_Dict[imgPath] = dist
    #print(entropy, DNNResult, lable, dist, layersHM[0][0][0], centroidHM[0][0][0])
    #print("entropy = ", entropy)
    if caseFile["datasetName"] == "FLD":
        entropy = entropy_FLD
    elif caseFile["datasetName"] == "HPD":
        entropy = entropy_HPD

    return entropy, DNNResult, None, lable, dist, layersHM


def putMask(imgPath, img, px, py):
    color = (240, 207, 137)
    thick = 23
    img = cv2.line(img, (px[0], py[0]), (px[6], py[6]), color, thick)  # N1 - N2
    img = cv2.line(img, (px[6], py[6]), (px[1], py[1]), color, thick)  # N2 - N3
    img = cv2.line(img, (px[0], py[0]), (px[1], py[1]), color, thick - 4)  # N1 - N3
    img = cv2.line(img, (px[0], py[0]), (px[4], py[4]), color, thick - 4)  # N1 - M1
    img = cv2.line(img, (px[0], py[0]), (px[5], py[5]), color, thick)  # N1 - M3
    img = cv2.line(img, (px[1], py[1]), (px[5], py[5]), color, thick - 4)  # N3 - M3
    img = cv2.line(img, (px[1], py[1]), (px[4], py[4]), color, thick)  # N3 - M1
    img = cv2.line(img, (px[5], py[5]), (px[2], py[2]), color, thick + 2)  # M3 - M2
    img = cv2.line(img, (px[2], py[2]), (px[4], py[4]), color, thick + 2)  # M2 - M1
    img = cv2.line(img, (px[4], py[4]), (px[3], py[3]), color, thick)  # M1 - M4
    img = cv2.line(img, (px[3], py[3]), (px[5], py[5]), color, thick)  # M4 - M3
    img = cv2.line(img, (px[4], py[4]), (px[5], py[5]), color, thick)  # M1 - M3
    label = labelImage(imgPath)
    thick = 4
    length = 35
    angle = 20
    print(label)
    if label == "BottomCenter":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-length-15, py[4]-angle), color, thick)# holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+length-25, py[5]-angle+5), color, thick)# holder (M1 + 15)
        img = img
    elif label == "BottomRight":
        img = cv2.line(img, (px[4], py[4]), (px[4] - length, py[4] - angle), color, thick)  # holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+35, py[5]-20), color, thick)# holder (M1 + 15)
    elif label == "BottomLeft":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-35, py[4]-20), color, thick)# holder (M1 + 15)
        img = cv2.line(img, (px[5], py[5]), (px[5] + length, py[5] - angle), color, thick)  # holder (M1 + 15)
    elif label == "MiddleRight":
        img = cv2.line(img, (px[4], py[4]), (px[4] - length, py[4] - angle), color, thick)  # holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]-35, py[5]-20), color, thick)# holder (M1 + 15)
    elif label == "MiddleLeft":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-35, py[4]-20), color, thick)# holder (M1 + 15)
        img = cv2.line(img, (px[5], py[5]), (px[5] + length, py[5] - angle), color, thick)  # holder (M1 + 15)
    elif label == "MiddleCenter":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-length-15, py[4]-angle-5), color, thick)# holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+length-10, py[5]-angle-5), color, thick)# holder (M1 + 15)
        img = img
    elif label == "TopLeft":
        # img = cv2.line(img, (px[4], py[4]), (px[4]-35, py[4]-20), color, thick)# holder (M1 + 15)
        img = cv2.line(img, (px[5], py[5]), (px[5] + length, py[5] - angle), color, thick)  # holder (M1 + 15)
    elif label == "TopRight":
        img = cv2.line(img, (px[4], py[4]), (px[4] - length, py[4] - angle), color, thick)  # holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+35, py[5]+20), color, thick)# holder (M1 + 15)
    elif label == "TopCenter":
        img = img
        # img = cv2.line(img, (px[4], py[4]), (px[4]-length-10, py[4]-angle-5), color, thick)# holder (M1 + 15)
        # img = cv2.line(img, (px[5], py[5]), (px[5]+length-10, py[5]-angle-5), color, thick)# holder (M1 + 15)
    return img

def putEyeglasses(imgPath, img, px, py):
    color = (240, 207, 137)
    thick = 2
    radius = 12
    if px[0] and py[0] is not None:
        img = cv2.circle(img, (px[0], py[0]), radius, color, thick)
    if px[1] and py[1] is not None:
        img = cv2.circle(img, (px[1], py[1]), radius, color, thick)
    if px[0] and py[0] and px[1] and py[1] is not None:
        img = cv2.line(img, (px[0] + radius, py[0]), (px[1] - radius, py[1]), color, thick)  # N1 - N2
    #img = cv2.line(img, (px[6], py[6]), (px[1], py[1]), color, thick)  # N2 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[1], py[1]), color, thick - 4)  # N1 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[4], py[4]), color, thick - 4)  # N1 - M1
    #img = cv2.line(img, (px[0], py[0]), (px[5], py[5]), color, thick)  # N1 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[5], py[5]), color, thick - 4)  # N3 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[4], py[4]), color, thick)  # N3 - M1
    #img = cv2.line(img, (px[5], py[5]), (px[2], py[2]), color, thick + 2)  # M3 - M2
    #img = cv2.line(img, (px[2], py[2]), (px[4], py[4]), color, thick + 2)  # M2 - M1
    #img = cv2.line(img, (px[4], py[4]), (px[3], py[3]), color, thick)  # M1 - M4
    #img = cv2.line(img, (px[3], py[3]), (px[5], py[5]), color, thick)  # M4 - M3
    #img = cv2.line(img, (px[4], py[4]), (px[5], py[5]), color, thick)  # M1 - M3
    return img

def putSunglasses(imgPath, img, px, py):
    color = (240, 207, 137)
    thick = 17
    radius = 7

    if px[0] and py[0] is not None:
        img = cv2.circle(img, (px[0], py[0]), radius, color, thick)
    if px[1] and py[1] is not None:
        img = cv2.circle(img, (px[1], py[1]), radius, color, thick)
    if px[0] and py[0] and px[1] and py[1] is not None:
        img = cv2.line(img, (px[0] + radius, py[0]), (px[1] - radius, py[1]), color, 2)  # N1 - N2
    #img = cv2.line(img, (px[7], py[7]), (px[8], py[8]), color, thick)  # N1 - N2
    #img = cv2.line(img, (px[6], py[6]), (px[1], py[1]), color, thick)  # N2 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[1], py[1]), color, thick - 4)  # N1 - N3
    #img = cv2.line(img, (px[0], py[0]), (px[4], py[4]), color, thick - 4)  # N1 - M1
    #img = cv2.line(img, (px[0], py[0]), (px[5], py[5]), color, thick)  # N1 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[5], py[5]), color, thick - 4)  # N3 - M3
    #img = cv2.line(img, (px[1], py[1]), (px[4], py[4]), color, thick)  # N3 - M1
    #img = cv2.line(img, (px[5], py[5]), (px[2], py[2]), color, thick + 2)  # M3 - M2
    #img = cv2.line(img, (px[2], py[2]), (px[4], py[4]), color, thick + 2)  # M2 - M1
    #img = cv2.line(img, (px[4], py[4]), (px[3], py[3]), color, thick)  # M1 - M4
    #img = cv2.line(img, (px[3], py[3]), (px[5], py[5]), color, thick)  # M4 - M3
    #img = cv2.line(img, (px[4], py[4]), (px[5], py[5]), color, thick)  # M1 - M3
    return img

def doTime(strW, start):
    print(strW, math.ceil((time.time() - start) / 60.0), " mins")
    return str(strW + str(math.ceil((time.time() - start) / 60.0)) + " mins")


def saveCP(res, path):
    with open(path, 'wb') as config_dictionary_file:
        # Step 3
        pickle.dump(res, config_dictionary_file)


def loadCP(path):
    with open(path, 'rb') as config_dictionary_file:
        res = pickle.load(config_dictionary_file)
    return res


def labelBIWI(txtPath):
    with open(txtPath) as f:
        lines = f.readlines()
        HP = lines[4]
    HP1 = float(HP.split(" ")[0])
    HP2 = float(HP.split(" ")[1])

    if HP2 < 27:
        lab = "Top"
    elif 27 <= HP2 <= 66:
        lab = "Middle"
    else:
        lab = "Bottom"

    if HP1 < 42:
        lab += "Left"
    elif 42 <= HP1 <= 68:
        lab += "Center"
    else:
        lab += "Right"
    return lab


def processBIWI(imgPath, dlibPath, newimgPath):
    img = cv2.imread(imgPath)
    # img = img[200:600, 100:400]
    # cv2.imwrite(newimgPath, img)
    face_detector = dlib.cnn_face_detection_model_v1(dlibPath)
    faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(faces) < 1:
        return False
    big_face = -np.inf
    mx, my, mw, mh = 0, 0, 0, 0
    for face in faces:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        if w * h > big_face:
            big_face = w * h
            mx, my, mw, mh = x, y, w, h
    sw_0 = max(mx - 25 // 2, 0)
    sw_1 = min(mx + mw + 25 // 2, new_img.shape[1])  # empirical
    sh_0 = max(my - 25 // 2, 0)
    sh_1 = min(my + mh + 25 // 2, new_img.shape[0])  # empirical
    assert sh_1 > sh_0
    assert sw_1 > sw_0
    big_face = new_img[sh_0:sh_1, sw_0:sw_1]
    new_img = cv2.resize(big_face, (128, 128), interpolation=cv2.INTER_CUBIC)
    x_data = new_img
    x_data = np.repeat(x_data[:, :, np.newaxis], 3, axis=2)
    img = x_data
    cv2.imwrite(newimgPath, img)
    return True


def processImage(imgPath, dlibPath):
    global counter
    img = cv2.imread(imgPath)
    npPath = join(dirname(imgPath), basename(imgPath).split(".png")[0] + ".npy")
    configFile = np.load(npPath, allow_pickle=True)
    labelFile = configFile.item()['label']

    face_detector = dlib.cnn_face_detection_model_v1(dlibPath)
    faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    # img = new_img
    # labelFile = np.array(label_arr)
    width = img.shape[1]
    height = img.shape[0]
    mouth = [6, 7, 8, 23, 24, 25, 26]
    mouth = [32, 34, 36, 49, 52, 55, 58]
    #righteye
    eyes = [69, 70, 34]

    p1x = p1y = p2x = p2y = p3x = p3y = None
    #kag_id_map = {
    #    70: "left_eye_center",
    #    69: "right_eye_center",
    #    43: "left_eye_inner_corner",
    #    46: "left_eye_outer_corner",
    #    40: "right_eye_inner_corner",
    #    37: "right_eye_outer_corner",
    #    23: "left_eyebrow_inner_end",
    #    27: "left_eyebrow_outer_end",
    #    22: "right_eyebrow_inner_end",
    #    18: "right_eyebrow_outer_end",
    #    34: "nose_tip",
    #    55: "mouth_left_corner",
    #    49: "mouth_right_corner",
    #    52: "mouth_center_top_lip",
    #    58: "mouth_center_bottom_lip"
    #}
    #for KP in mouth:
    for KP in eyes:
        if KP in labelFile:
            x_p = labelFile[KP][0]
            y_p = img.shape[0] - labelFile[KP][1]
            px = x_p
            py = y_p

            if KP == 69:
                p1x = px
                p1y = py
            elif KP == 70:
                p2x = px
                p2y = py
            elif KP == 34:
                p3x = px
                p3y = py
            #if KP == 32:
            #    p1x = px
            #    p1y = py
            #elif KP == 36:
            #    p2x = px
            #    p2y = py
            #elif KP == 34:
            #    p7x = px
            #    p7y = py
            #elif KP == 58:
            #    p3x = px
            #    p3y = py
            #elif KP == 52:
            #    p4x = px
            #    p4y = py
            #elif KP == 49:
            #    p5x = px
            #    p5y = py
            #elif KP == 55:
            #    p6x = px
            #    p6y = py
    # new_img = putMask(imgPath, img, [p1x, p2x, p3x, p4x, p5x, p6x, p7x], [p1y, p2y, p3y, p4y, p5y, p6y, p7y])
    #new_img = putEyeglasses(imgPath, img, [p1x, p2x, p3x], [p1y, p2y, p3y])
    #new_img = putSunglasses(imgPath, img, [p1x, p2x, p3x], [p1y, p2y, p3y])
    #cv2.imwrite(imgPath, new_img)
    #img = new_img
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return
    if len(faces) < 1:
        return False, None, None
    big_face = -np.inf
    mx, my, mw, mh = 0, 0, 0, 0
    for face in faces:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        if w * h > big_face:
            big_face = w * h
            mx, my, mw, mh = x, y, w, h
    sw_0 = max(mx - 25 // 2, 0)
    sw_1 = min(mx + mw + 25 // 2, new_img.shape[1])  # empirical
    sh_0 = max(my - 25 // 2, 0)
    sh_1 = min(my + mh + 25 // 2, new_img.shape[0])  # empirical
    assert sh_1 > sh_0
    assert sw_1 > sw_0
    label_arr = []
    iee_labels = [18, 22, 23, 27, 28, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 69, 43, 44, 45, 46, 47, 48, 70, 49, 52,
                  55, 58]
    for ky in iee_labels:
        if ky in labelFile:
            coord = [labelFile[ky][0], new_img.shape[0] - labelFile[ky][1]]
            label_arr.append(coord)  # (-1,-1) means the keypoint is invisible
        else:
            label_arr.append([0, 0])  # label does not exist
    new_label = np.zeros_like(np.array(label_arr))
    new_label[:, 0] = np.array(label_arr)[:, 0] - sw_0
    new_label[:, 1] = np.array(label_arr)[:, 1] - sh_0
    new_label[new_label < 0] = 0
    new_label[np.array(label_arr)[:, 0] == -1, 0] = -1
    new_label[np.array(label_arr)[:, 1] == -1, 1] = -1
    big_face = new_img[sh_0:sh_1, sw_0:sw_1]
    width_resc = float(128) / big_face.shape[0]
    height_resc = float(128) / big_face.shape[1]
    new_label2 = np.zeros_like(new_label)
    new_label2[:, 0] = new_label[:, 0] * width_resc
    new_label2[:, 1] = new_label[:, 1] * height_resc
    labelFile = new_label2


    new_img = cv2.resize(big_face, (128, 128), interpolation=cv2.INTER_CUBIC)
    #print(new_img.shape)
    x_data = new_img
    x_data = np.repeat(x_data[:, :, np.newaxis], 3, axis=2)
    img = x_data
    cv2.imwrite(imgPath, img)
    label = labelImage(imgPath)
    label_dir = os.path.join(os.path.dirname(os.path.dirname(imgPath)), label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    newImgPath = os.path.join(label_dir, "S" + str(counter) + ".png")
    shutil.copy(imgPath.split(".png")[0] + ".npy", os.path.join(os.path.dirname(imgPath), "S" + str(counter) + ".npy"))
    cv2.imwrite(newImgPath, img)
    counter += 1
    return True, new_img, new_label2


def labelImage(imgPath):
    margin1 = 10.0
    margin2 = -10.0
    margin3 = 10.0
    margin4 = -10.0
    configPath = join(dirname(imgPath), basename(imgPath).split(".png")[0] + ".npy")
    configFile = np.load(configPath, allow_pickle=True)
    configFile = configFile.item()
    HP1 = configFile['config']['head_pose'][0]
    HP2 = configFile['config']['head_pose'][1]
    originalDst = None
    if HP1 > margin1:
        if HP2 > margin3:
            originalDst = "BottomRight"
        elif HP2 < margin4:
            originalDst = "BottomLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "BottomCenter"
    elif HP1 < margin2:
        if HP2 > margin3:
            originalDst = "TopRight"
        elif HP2 < margin4:
            originalDst = "TopLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "TopCenter"
    elif margin2 <= HP1 <= margin1:
        if HP2 > margin3:
            originalDst = "MiddleRight"
        elif HP2 < margin4:
            originalDst = "MiddleLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "MiddleCenter"
    if originalDst is None:
        print("cannot label img:", imgPath)
    return originalDst


def generateClusterImage(clusterData, clusterID, SimDataPath):
    cam_dir = [(0, 0), (0, 0), (0, 0)]
    cam_loc = [(0, 0), (0, 0), (0, 0)]
    lamp_loc = [(0, 0), (0, 0), (0, 0)]
    lamp_col = [(0, 0), (0, 0), (0, 0)]
    lamp_dir = [(0, 0), (0, 0), (0, 0)]
    lamp_eng = [(0, 0)]
    head = [False, (0, 0), False, (0, 0), False, (0, 0)]
    for param in clusterData[clusterID]:
        if param.startswith("cam_look"):
            if param.endswith("0"):
                cam_dir[0], cam_dir[1] = getParams(-0.087, 0.87, clusterData[clusterID][param], False)
            if param.endswith("1"):
                cam_dir[2], cam_dir[3] = getParams(-4.29, -0.87, clusterData[clusterID][param], False)
            if param.endswith("2"):
                cam_dir[4], cam_dir[5] = getParams(-1.39, 0.69, clusterData[clusterID][param], False)
        if param.startswith("cam_loc"):
            if param.endswith("0"):
                cam_loc[0], cam_loc[1] = getParams(0.29, 0.38, clusterData[clusterID][param], False)
            if param.endswith("1"):
                cam_loc[2], cam_loc[3] = getParams(-5.71, -5.00, clusterData[clusterID][param], False)
            if param.endswith("2"):
                cam_loc[4], cam_loc[5] = getParams(14.70, 16.60, clusterData[clusterID][param], False)
        if param.startswith("lamp_loc"):
            if param.endswith("0"):
                lamp_loc[0], lamp_loc[1] = getParams(-29.75, 30.87, clusterData[clusterID][param], False)
            if param.endswith("1"):
                lamp_loc[2], lamp_loc[3] = getParams(-62.07, 46.82, clusterData[clusterID][param], False)
            if param.endswith("2"):
                lamp_loc[4], lamp_loc[5] = getParams(-27.41, 40.93, clusterData[clusterID][param], False)
        if param.startswith("lamp_color"):
            if param.endswith("0"):
                lamp_col[0], lamp_col[1] = getParams(-4.29, 1.0, clusterData[clusterID][param], False)
            if param.endswith("1"):
                lamp_col[2], lamp_col[3] = getParams(-1.39, 1.0, clusterData[clusterID][param], False)
            if param.endswith("2"):
                lamp_col[4], lamp_col[5] = getParams(-50, 50, clusterData[clusterID][param], False)
        if param.startswith("lamp_direct"):
            if param.endswith("0"):
                lamp_dir[0], lamp_dir[1] = getParams(0.29, 0.87, clusterData[clusterID][param], False)
            if param.endswith("1"):
                lamp_dir[2], lamp_dir[3] = getParams(-5.00, -0.87, clusterData[clusterID][param], False)
            if param.endswith("2"):
                lamp_dir[4], lamp_dir[5] = getParams(0.69, 14.70, clusterData[clusterID][param], False)
        if param.startswith("lamp_energy"):
            lamp_eng[0], lamp_eng[1] = getParams(-0.08, 1.0, clusterData[clusterID][param], False)
        if param.startswith("head"):
            if param.endswith("0"):
                head[0], head[1] = getParams(-28.88, 8.9, clusterData[clusterID][param], False)
            if param.endswith("1"):
                head[2], head[3] = getParams(-86.87, 72.91, clusterData[clusterID][param], False)
            if param.endswith("2"):
                head[4], head[5] = getParams(-20.18, 36.26, clusterData[clusterID][param], False)
    m = random.randint(0, len(pose_file) - 1)
    # if m == 0:
    #    m = 1
    # else:
    #    m = 0
    # generator = IEEKPgenerator(model_folder, pose_folder, label_folder)
    # generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
    #                                         cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
    #                                         label_file[m])


def generateRandomImage(SimDataPath):
    cam_dir = [0, 0, 0]
    cam_loc = [0, 0, 0]
    lamp_loc = [0, 0, 0]
    lamp_col = [0, 0, 0]
    lamp_dir = [0, 0, 0]
    lamp_eng = [0]
    head = [0, 0, 0]
    for param in param_list:
        if param.startswith("cam_look"):
            if param.endswith("0"):
                cam_dir[0] = getParams(-0.105, -0.085, [-0.087, 0.87], True)
            if param.endswith("1"):
                cam_dir[1] = getParams(-4.674, -4.29, [-4.29, -0.87], True)
            if param.endswith("2"):
                cam_dir[2] = getParams(-1.699, -1.275, [-1.39, 0.69], True)
        if param.startswith("cam_loc"):
            if param.endswith("0"):
                cam_loc[0] = getParams(0.261, 0.293, [0.261, 0.293], True)
            if param.endswith("1"):
                cam_loc[1] = getParams(-5.351, -5.00, [-5.351, -5.00], True)
            if param.endswith("2"):
                cam_loc[2] = getParams(14.445, 14.869, [14.445, 14.869], True)
        if param.startswith("lamp_loc"):
            if param.endswith("0"):
                lamp_loc[0] = getParams(0.361, 0.381, [-29.75, 30.87], True)
            if param.endswith("1"):
                lamp_loc[1] = getParams(-5.72, 5.61, [-62.07, 46.82], True)
            if param.endswith("2"):
                lamp_loc[2] = getParams(16.543, 16.644, [-27.41, 40.93], True)
        if param.startswith("lamp_color"):  # FIXME -- constant
            if param.endswith("0"):
                lamp_col[0] = getParams(1.0, 1.0, [-4.29, 1.0], True)
            if param.endswith("1"):
                lamp_col[1] = getParams(1.0, 1.0, [-1.39, 1.0], True)
            if param.endswith("2"):
                lamp_col[2] = getParams(1.0, 1.0, [-50, 50], True)
        if param.startswith("lamp_direct"):  # FIXME -- constant
            if param.endswith("0"):
                lamp_dir[0] = getParams(0.873, 0.873, [0.29, 0.87], True)
            if param.endswith("1"):
                lamp_dir[1] = getParams(-0.873, -0.873, [-5.00, -0.87], True)
            if param.endswith("2"):
                lamp_dir[2] = getParams(0.698, 0.698, [0.69, 14.70], True)
        if param.startswith("lamp_energy"):  # FIXME -- constant
            lamp_eng[0] = getParams(50.0, 500.0, [-0.08, 1.0], True)
        if param.startswith("head"):
            if param.endswith("0"):
                head[0] = getParams(-41.86, 36.87, [-28.88, 8.9], True)
            if param.endswith("1"):
                head[1] = getParams(-79.86, 75.13, [-86.87, 72.91], True)
            if param.endswith("2"):
                head[2] = getParams(-64.30, 51.77, [-20.18, 36.26], True)
    m = random.randint(0, len(pose_file) - 1)
    # command = "/Applications/Blender.app/Contents/MacOS/blender --background --python ./ieekeypoints.py -- " \
    #          "--model_folder {} --label_folder {} --pose_folder {} --head {} --lamp_dir {} --lamp_eng {} --lamp_col {} " \
    #          "--lamp_loc {} --cam_loc {} --cam_dir {} --pose_file {} --label_file {} --model_file {} --SimDataPath {} " \
    #          "--m {}".format(model_folder, label_folder, pose_folder, head, lamp_dir, lamp_eng, lamp_col, lamp_loc,
    #                                         cam_loc, cam_dir, pose_file, label_file, model_file, SimDataPath, m)
    # print (command)
    # subprocess.call(command.split(" "), cwd=IEE_SIM_abs_path)
    blenderPath = "/Applications/Blender.app/Contents/MacOS/blender"
    filePath = DIR + "ieekeypoints.py"
    imgPath = join(SimDataPath, processX([cam_dir[0], cam_dir[0], cam_dir[0], cam_loc[0], cam_loc[0], cam_loc[0]]))
    process = subprocess.run(
        [str(blenderPath), "--background", "--verbose", str(0), "--python", str(filePath), "--", "--path",
         str(path), "--model_folder",
         str(model_folder), "--label_folder", str(label_folder), "--pose_folder", str(pose_folder), "--pose_file",
         str(pose_file[m]), "--label_file", str(label_file[m]), "--model_file", str(model_file[m]), "--imgPath",
         # str(imgPath)], stdout=subprocess.PIPE)
         str(imgPath)], stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
    subprocess.Popen(
        ["/Applications/Blender.app/Contents/MacOS/blender --background --verbose 0 --python ./ieekeypoints.py -- " \
         "--model_folder {} --label_folder {} --pose_folder {} --head {} --lamp_dir {} --lamp_eng {} --lamp_col {} " \
         "--lamp_loc {} --cam_loc {} --cam_dir {} --pose_file {} --label_file {} --model_file {} --SimDataPath {} " \
         "--m {}".format(model_folder, label_folder, pose_folder, head, lamp_dir, lamp_eng, lamp_col, lamp_loc,
                         cam_loc, cam_dir, pose_file, label_file, model_file, SimDataPath, m)],
        stdout=subprocess.PIPE)

    # generator = ieeKP.IEEKPgenerator(model_folder, pose_folder, label_folder)
    # imgPath = generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
    #                                         cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
    #                                         label_file[m])
    # return imgPath


def evalImages(clsData, caseFile):
    global layer
    for member in clsData['clusters'][clusterID]['members']:
        fileName = member.split("_")[1]
        data = np.load(caseFile["testDataNpy"], allow_pickle=True)
        data = data.item()
        configData = data['config'][int(fileName) - 1]
        x = [configData['cam_look_direction'][0], configData['cam_look_direction'][1],
             configData['cam_look_direction'][2], configData['cam_loc'][0], configData['cam_loc'][1],
             configData['cam_loc'][2], configData['lamp_loc_Lamp'][0], configData['lamp_loc_Lamp'][1],
             configData['lamp_loc_Lamp'][2], configData['lamp_color_Lamp'][0], configData['lamp_color_Lamp'][1],
             configData['lamp_color_Lamp'][2], configData['lamp_direct_xyz_Lamp'][0],
             configData['lamp_direct_xyz_Lamp'][1], configData['lamp_direct_xyz_Lamp'][2],
             configData['lamp_energy_Lamp'], configData["head_pose"][0], configData["head_pose"][1],
             configData["head_pose"][2]]
        imgPath, _, _, _ = generateAnImage(x, caseFile)
        processImage(imgPath + ".png", join(caseFile["outputPath"],
                                            "IEEPackage/clsdata/mmod_human_face_detector.dat"))
        layersHM, entropy = generateHeatMap(imgPath + ".png", caseFile["DNN"],
                                            caseFile["datasetName"],
                                            caseFile["outputPath"], False, None, None,
                                            caseFile["imgExt"], None)
        newImgPath = join(caseFile["DataSetsPath"], "TestSet_Backup", str(int(fileName) - 1) + ".png")
        print()
        layersHM2, entropy = generateHeatMap(newImgPath, caseFile["DNN"],
                                             caseFile["datasetName"],
                                             caseFile["outputPath"], False, None, None,
                                             caseFile["imgExt"], None)
        dist = doDistance(centroidHM,
                          layersHM[layer],
                          "Euc")

        dist2 = doDistance(centroidHM,
                           layersHM2[layer],
                           "Euc")
        print(1 - (clusterRadius / dist))
        print(1 - (clusterRadius / dist2))


def getClustersData(caseFile, heatMapDistanceExecl, clusterID):
    centroidRadius, centroidHMs = getClusterData(caseFile, heatMapDistanceExecl)
    return centroidRadius[clusterID], centroidHMs[clusterID]


if __name__ == '__main__':
    casePath = join(outputPath, "T/caseFile_None.pt")
    caseFile = torch.load(casePath)
    for clusterID in [1]:
        # Get cluster's radius and centroid's heatmap
        # clusterRadius, centroidHM = getClustersData(
        #    join(caseFile["filesPath"], "Heatmaps", caseFile["selectedLayer"] + "HMDistance.xlsx"), clusterID)
        outPath = join(caseFile["filesPath"], "GeneratedImages", str(clusterID))
        # problem = HUDDProblem(outPath, clusterRadius, centroidHM, caseFile)
        problem = HUDDProblem(None, None, caseFile)
        algorithm = NSGA2(pop_size=10, sampling=get_sampling("bin_random"), crossover=get_crossover("bin_two_point"),
                          mutation=get_mutation("bin_bitflip"), eliminate_duplicates=True)
        res = minimize(problem, algorithm, ('n_gen', 200), seed=1, verbose=False)
        plot = Scatter()
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(res.F, color="red")
        plot.show()
    # for component in components:
    #    clsData = torch.load()
    # clustersData = np.load(join(outputPath, component, "clustersParamData.npy"), allow_pickle=True)
    # clusterData = clustersData.item()
    #    for clusterID in clsData:
    #        SimDataPath = join(outputPath, component, "SimData", "SimulatorData", "Cluster_"+str(clusterID), "Data")
    #        counter = 0
    #        while counter < U:
    #            m = 0
    # generateParamsImage(clusterData, clusterID, SimDataPath)
    #            generateRandomImage(SimDataPath)
    #            counter += 1
