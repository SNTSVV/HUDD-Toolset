
from os.path import isfile, join, basename, dirname
import sys
import argparse
import os
from os import listdir
sys.path.insert(1, './')
import ieesimulator as ieesim
#import numpy as np

def generate(data):
    pose_file, model_file, label_file = data[0], data[1], data[2]
    width, height = data[3], data[4] #376
    dst_datafolder = data[5] #240
    imitator = ieesim.IEEImitator(model_file,label_file,pose_file,\
               width,height, dst_datafolder=dst_datafolder)
    imitator.create_samples()
    return

def get_files(folder, extx=".csv"):
    all_files = []
    for f in listdir(folder):
        xf = join(folder, f)
        if isfile(xf) and xf.endswith(extx):
            all_files.append(xf)
    return all_files


class IEEKPgenerator(object):
    def __init__(self, model_folder, pose_folder, label_folder):
        self.model_folder = model_folder
        self.pose_folder = pose_folder
        self.label_folder = label_folder
        self.model_pose_labels = self.get_model_pose_labels()
        return

    #add a shuffle to the file lists?
    def get_model_pose_labels(self):
        all_mod_files = get_files(self.model_folder, ".mhx2")
        #utils.print("all_mod_files: ", len(all_mod_files))
        all_pose_files = get_files(self.pose_folder, ".csv")
        def match_pose_label(all_mod_files, label_folder):
            model_label_pairs = []
            for model_file in all_mod_files:
                model_who = os.path.basename(model_file)
                model_who = model_who.split(".mhx2")[0]
                label_file = label_folder+"/"+model_who.lower()+"_label.npy"
                #print(label_folder, model_who)
                if isfile(label_file):
                    #utils.print("matched label file: ", label_file)
                    model_label_pairs.append([model_file,label_file])
            #print(model_label_pairs)
            return  model_label_pairs

        model_label_pairs = match_pose_label(all_mod_files, self.label_folder)

        num_to_simu = min(len(all_pose_files), len(model_label_pairs))

        if num_to_simu < 1:
            return None

        model_label_pairs = model_label_pairs[:num_to_simu]
        model_files, label_files = zip(*model_label_pairs)
        all_pose_files = all_pose_files[:num_to_simu]
        
        return zip(all_pose_files, model_files, label_files)

    def generate_with_single_processor(self, img_width, img_height, imgPath, pose_file, model_file, label_file, x=None):
        if not self.model_pose_labels:
            print("no file exists")
            return
        faceControl = False
        if pose_file is not None:
            pose_file = pose_file + ".csv" #Gle
            faceControl = True
        if model_file is not None:
            model_file = model_file + ".mhx2" #Aaf01_o
            faceControl = True
        if label_file is not None:
            label_file = label_file + "_label.npy" #aaf01_o
            faceControl = True
        if faceControl:
            #print("INFO: using 3D model: ", model_file)
            #print("INFO: using label file: ", label_file)
            #print("INFO: using pose file: ", pose_file)
            if x is None:
                x = str(basename(imgPath).split(".png")[0]).split("_")
                xfloat = list()
                for x1 in x:
                    xfloat.append(float(x1))
                x = xfloat

            imitator = ieesim.IEEImitator(model_file, label_file, pose_file, img_width, img_height,
                                          dirname(imgPath), False)
            imitator.create_samples_allParams((x[0], x[1], x[2]), (x[3], x[4], x[5]), (x[6], x[7], x[8]),
                                              (x[9], x[10], x[11]),
                                              1, imgPath)
        else:
            for pose_file, model_file, label_file in self.model_pose_labels:
                #print("INFO: using 3D model: ", model_file)
                #print("INFO: using label file: ", label_file)
                #print("INFO: using pose file: ", pose_file)
                imitator = ieesim.IEEImitator(model_file, label_file, pose_file, img_width, img_height, dirname(imgPath), False)
                imitator.create_samples(head, lamp_dir, lamp_col, lamp_loc, lamp_eng, cam_loc, cam_dir, 1e9)

    def generate_with_multi_processor(self, img_width, img_height, dst_datafolder=None):
        if not self.model_pose_labels:
            print("no file exists")
            return

        import multiprocessing
        from multiprocessing import Pool

        cpu_count = multiprocessing.cpu_count()

        def form_parameters(model_pose_labels, img_width, img_height, dst_datafolder):
            pose_file, model_file, label_file = zip(*model_pose_labels)
            width_arr = [img_width]*len(pose_file)
            height_arr = [img_height]*len(pose_file)
            dsts = [dst_datafolder]*len(pose_file)
            all_paras = zip(pose_file,model_file,label_file,width_arr,height_arr,dsts)
            return all_paras

        all_paras = form_parameters(self.model_pose_labels, img_width, img_height, dst_datafolder)
        all_paras = list(all_paras)

        for idx in range(len(all_paras)):
            print(all_paras[idx])

        use_cpu = min(len(all_paras), cpu_count-1)
        use_cpu = max(1,use_cpu)

        print("INFO: ", cpu_count, " CPUs exists.", use_cpu, "will be used.")
        pool = Pool(processes=use_cpu)
        
        pool.map(generate, all_paras)

if __name__ == '__main__':
    #model_folder = "./mhx2"
    #label_folder = "./newlabel3d"
    #pose_folder = "./pose"
    width = 752//2 #376
    height = 480//2 #240
    parser = argparse.ArgumentParser(description='DNN debugger')
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    path = argv[1]
    model_folder = argv[3]
    label_folder = argv[5]
    pose_folder = argv[7]
    pose_file = argv[9]
    label_file = argv[11]
    model_file = argv[13]
    imgPath = argv[15]
    print("Generating an image", basename(imgPath))
    generator = IEEKPgenerator(join(path, model_folder), join(path, pose_folder),
                               join(path, label_folder))

    #generator.generate_with_multi_processor(width, height)
    #generator.generate_with_single_processor(width, height)
    generator.generate_with_single_processor(width, height, imgPath, join(path, pose_folder, pose_file),
                                             join(path, model_folder, model_file),
                                             join(path, label_folder, label_file))
    #generator.generate_with_multi_processor(width, height,"./test_img/")
    #generator.generate_with_single_processor(width, height, "./test_img/")














    