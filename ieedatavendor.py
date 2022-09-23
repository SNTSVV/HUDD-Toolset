from imports import Variable, tqdm, pd, np, cv2, os, dlib, glob, exists, join, isfile, makedirs

def ensure_folder(folder_name):
    if not exists(folder_name):
        os.makedirs(folder_name)
    return


class IEEDataVendor(object):
    def __init__(self, syth_fold_path, kagl_fold_path, face_detector_path, syth_tol=(25,25), real_tol=(10,10), img_size=128):
        self.face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
        self.syth_fold_path = syth_fold_path
        self.kagl_fold_path = kagl_fold_path
        self.img_size = img_size #the output image size 
        self.iee_labels = self.get_iee_labels()
        self.syth_wtol, self.syth_htol = syth_tol
        self.real_wtol, self.real_htol = real_tol

    
    def get_iee_labels(self):
        labels = [18, 22, 23, 27, 28, 31, 32, 34, 36, 37, 38,39, 40, 41, 42, 69, 43, 44, 45, 46, 47, 48, 70,49,52,55,58]
        return labels

    def get_imgs(self, folder, ext="png"):
        folder_f = folder+"/*."+ext
        f_list = glob.glob(folder_f)
        return f_list

    def save_data(self, data, label, config, dst, file_name):
        ensure_folder(dst)
        dataset = {"data":data, "label":label, "config":config}
        np.save(dst+"/"+file_name+".npy", dataset)
        return

    def split_data(self, data, label, dst, file_name, ratio=0.7):
        ensure_folder(dst)
        r_idx = np.random.permutation(data.shape[0])
        data = data[r_idx]
        label = label[r_idx]

        part_len = int(data.shape[0]*ratio)

        dataset = {"data":data[:part_len], "label":label[:part_len]}
        np.save(dst+"/"+file_name+"_train.npy", dataset)

        dataset = {"data":data[part_len:], "label":label[part_len:]}
        np.save(dst+"/"+file_name+"_test.npy", dataset)

        return


    def get_data_list(self, folder, shuffle=True):
        subfolders = [ f.path for f in os.scandir(folder) if f.is_dir()]
        if shuffle:
            np.random.shuffle(subfolders)
        return subfolders

    def get_id2kaggle_map(self):
        kag_id_map = {
        70:"left_eye_center",
        69:"right_eye_center",
        43:"left_eye_inner_corner",
        46:"left_eye_outer_corner",
        40:"right_eye_inner_corner",
        37:"right_eye_outer_corner",
        23:"left_eyebrow_inner_end",
        27:"left_eyebrow_outer_end",
        22:"right_eyebrow_inner_end",
        18:"right_eyebrow_outer_end",
        34:"nose_tip",
        55:"mouth_left_corner",
        49:"mouth_right_corner",
        52:"mouth_center_top_lip",
        58:"mouth_center_bottom_lip"
        }
        return kag_id_map

    def load_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def get_kaggle_labels(self, kag_df):
        id2kag = self.get_id2kaggle_map()
        (row, col) = kag_df.shape
        label_arr = np.zeros((row, len(self.iee_labels), 2))

        for idx, ky in enumerate(self.iee_labels):
            if ky in id2kag:
                ky_name = id2kag[ky]
                coords = kag_df[[ky_name+"_x", ky_name+"_y"]].values
                label_arr[:,idx,:]=coords
        return label_arr

    def get_kaggle_data(self):
        df = pd.read_csv(self.kagl_fold_path)
        df = df.fillna(-1)
        df['Image'] = df['Image'].apply(lambda img:  np.fromstring(img, sep = ' '))
        x_data = np.vstack(df['Image'].values)
        x_data = x_data.astype(np.uint8)
        #x_data = x_data / 255.   # scale pixel values to [0, 1]
        x_data = x_data.reshape(-1, 1, 96, 96) # return each images as 1 x 96 x 96
        y_data = self.get_kaggle_labels(df)
        #print("kaggle y_data: ", y_data.shape)
        return x_data, y_data
    
    # get_real_data wraps get_kaggle_data, and converts the data size to (1,128,128)
    def get_real_data(self):
        x_data, y_data = self.get_kaggle_data()
        new_x_data = []
        new_y_data = []
        noface_num = 0
        for idx in tqdm(range(x_data.shape[0])):
            img_data = x_data[idx,:]
            img_data = cv2.cvtColor(img_data[0,:], cv2.COLOR_GRAY2BGR)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY) #FIXME

            label_data = y_data[idx,:]
            #est biggest face
            img_data = self.get_the_biggest_face(img_data, label_data, self.real_wtol, self.real_htol)
            if not img_data:
                noface_num += 1
                #print("cannot find face in img: ", idx, " skip...", noface_num)
                continue
            new_data, new_label = self.resize(img_data[0], img_data[1], self.img_size)
            new_y_data.append(new_label)
            new_x_data.append(new_data)
        print("INFO: ", noface_num, " imgs cannot find face at kaggle dataset")

        return (np.array(new_x_data), np.array(new_y_data))

    def get_syth_label(self, apng, img_height):
        label_file = apng.split(".png")[0]+".npy"
        if not isfile(label_file):
            return None

        data = np.load(label_file, allow_pickle=True)
        label_data = data.item()["label"]
        label_arr = []
        for ky in self.iee_labels:
            if ky in label_data:
                coord = [label_data[ky][0], img_height-label_data[ky][1]]
                label_arr.append(coord) #(-1,-1) means the keypoint is invisible
            else:
                label_arr.append([0,0]) # label does not exist
        return np.array(label_arr), data.item()["config"]

    def get_the_biggest_face(self, img_data, img_label, w_tol, h_tol):
        faces = self.face_detector(img_data,1)
        if len(faces) < 1:
            return None

        big_face = -np.inf
        mx, my, mw, mh = 0, 0, 0, 0
        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            if w*h > big_face:
                big_face = w*h
                mx, my, mw, mh = x,y,w,h

        sw_0 = max(mx-w_tol//2,0)
        sw_1 = min(mx+mw+w_tol//2, img_data.shape[1]) #empirical

        sh_0 = max(my-h_tol//2,0)
        sh_1 = min(my+mh+h_tol//2, img_data.shape[0]) #empirical

        assert sh_1 > sh_0
        assert sw_1 > sw_0

        big_face = img_data[sh_0:sh_1, sw_0:sw_1]

        new_label = np.zeros_like(img_label)
        new_label[:,0] = img_label[:,0]-sw_0
        new_label[:,1] = img_label[:,1]-sh_0

        new_label[new_label<0] = 0
        new_label[img_label[:,0]==-1,0] = -1
        new_label[img_label[:,1]==-1,1] = -1 #FIXme : new_label[img_label[:,0]==-1] = -1

        return (big_face, new_label)

    def resize(self, img_face, img_label, target_size):
        (ori_w, ori_h) = img_face.shape
        #print("resize origin: ", img_face.shape, img_label.shape)

        new_img = cv2.resize(img_face, (target_size,target_size), interpolation=cv2.INTER_CUBIC)
        width_resc = float(target_size)/ori_w
        height_resc = float(target_size)/ori_h

        new_label = np.zeros_like(img_label)

        new_label[:,0] = img_label[:,0]*width_resc
        new_label[:,1] = img_label[:,1]*height_resc
        return new_img, new_label

    #@folder_list: must be a list
    def get_syth_data(self, folder_list):
        x_data = []
        y_data = []
        z_data = []
        print(folder_list)
        noface_num = 0
        for onefolder in tqdm(folder_list):
            all_pngs = self.get_imgs(onefolder)
            for apng in all_pngs:
                img = self.load_img(apng)
                label_data, config_data = self.get_syth_label(apng, img.shape[0])
                big_face = self.get_the_biggest_face(img, label_data, self.syth_wtol, self.syth_htol)

                if not big_face:
                    noface_num += 1
                    print("cannot find face in img: ", apng, " skip...", noface_num)
                    continue

                new_data, new_label = self.resize(big_face[0], big_face[1], self.img_size)
                y_data.append(new_label)
                x_data.append(new_data)
                z_data.append(config_data)

            print("INFO: ", noface_num, " imgs cannot find face at: ", onefolder)
        return (np.array(x_data), np.array(y_data), np.array(z_data))

    #generate:train data, test_data, real_data
    def generate_data(self, shuffle=True, test=True, train=True):
        data_list = self.get_data_list(self.syth_fold_path, shuffle=shuffle)
        print(data_list)
        if test:
            test_set = self.get_syth_data(data_list) #the input should be a list
            print("---test_set----",test_set[0].shape, test_set[1].shape, test_set[2].shape)
            return (None, test_set, None)
        elif test and train:
            test_set = self.get_syth_data([data_list[0]]) #the input should be a list
            print("---test_set----",test_set[0].shape, test_set[1].shape, test_set[2].shape)
            train_set = self.get_syth_data(data_list[1:])
            print("---train_set----",train_set[0].shape, train_set[1].shape, train_set[2].shape)
            return (train_set, test_set, None)
        #real_set = self.get_real_data()
        #print("----real_set---",real_set[0].shape, real_set[1].shape)
        #real_set = None
        #return (train_set, test_set, real_set)

    def save_evidence(self, imgs, labels, dst, red_factor=20):
        assert imgs.shape[0] == labels.shape[0]

        def update(img, x_t, y_t):
            (height, width, c) = img.shape
            for idx in [-1,0,1]:
                tx = max(min(x_t+idx, width-1),0)
                for jdx in [-1,0,1]:
                    ty = max(min(y_t+jdx, height-1),0)
                    if width>ty > 0 and height>tx >0:
                        img[ty,tx, 0] = 0
                        img[ty,tx, 1] = 0
                        img[ty,tx, 2] = 255
            return img

        for idx in tqdm(range(imgs.shape[0]//red_factor)):
            img = imgs[idx,:]
            img = np.repeat(img[ :, :, np.newaxis], 3, axis=2)
            label = labels[idx,:]
            for jdx in range(label.shape[0]):
                x_t = int(label[jdx][0]+0.5)
                y_t = int(label[jdx][1]+0.5)
                if x_t>0 and y_t>0:
                    img = update(img, x_t, y_t)

            file_name = dst+"/"+str(idx)+".png"
            ensure_folder(dst)
            cv2.imwrite(file_name,img)


def generate_data(syth_fold_path, kaggle_fold_path, face_detector_path, dst_folder):
    print("Generating data")
    vendor = IEEDataVendor(syth_fold_path, kaggle_fold_path, face_detector_path)
    print("Vendor is ready")
    train = False
    test = True
    (train_set, test_set, real_set) = vendor.generate_data(True, test, train) #TestSet only
    print("Generating test with length", len(test_set[0]))
    #vendor.save_evidence(test_set[0], test_set[1], dst_folder+"/test_env")
    #vendor.save_evidence(real_set[0], real_set[1], dst_folder+"/real_env")
    #vendor.save_evidence(train_set[0], train_set[1], dst_folder+"/train_env")
    vendor.save_data(test_set[0], test_set[1], test_set[2], dst_folder, "ieetest")
    #vendor.save_data(real_set[0], real_set[1], dst_folder, "ieereal")
    if train:
        part_len = int(train_set[0].shape[0] * 0.7)
        print("Generating train with length,", len(train_set[0][:part_len]))
        vendor.save_data(train_set[0][:part_len], train_set[1][:part_len], train_set[2][:part_len], dst_folder, "ieetrain")
        print("Generating improveset with length", len(train_set[0][part_len:]))
        vendor.save_data(train_set[0][part_len:], train_set[1][part_len:], train_set[2][part_len:], dst_folder, "ieeimprove")

    return

def exportIEEImages(dataSet, originalDst, prefix, mainCounter):
    if not exists(originalDst):
        makedirs(originalDst)
    if mainCounter is None:
        mainCounter = 1
    print(tqdm(dataSet))
    for (inputs, cp_labels) in tqdm(dataSet):
        inputs = Variable(inputs)
        inputs_cpu = inputs.cpu()
        inputs_cpu = inputs_cpu.detach().numpy()
        num_sample = inputs_cpu.shape[0]
        for idx in range(num_sample):
            img = inputs_cpu[idx] * 255.
            img = img[0, :]
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            file_name = join(originalDst, prefix + str(mainCounter) + ".png")
            if not exists(originalDst):
                os.mkdir(originalDst)
            cv2.imwrite(file_name, img)
            mainCounter += 1

def labelHPDimages(npPath, dataSet, originalDst, prefix, mainCounter):
        x = np.load(npPath, allow_pickle=True)
        x = x.item()["config"]
        TC = join(originalDst, "TopCenter")
        if not exists(TC):
            makedirs(TC)
        TR = join(originalDst, "TopRight")
        if not exists(TR):
            makedirs(TR)
        TL = join(originalDst, "TopLeft")
        if not exists(TL):
            makedirs(TL)
        BC = join(originalDst, "BottomCenter")
        if not exists(BC):
            makedirs(BC)
        BL = join(originalDst, "BottomLeft")
        if not exists(BL):
            makedirs(BL)
        BR = join(originalDst, "BottomRight")
        if not exists(BR):
            makedirs(BR)
        MR = join(originalDst, "MiddleRight")
        if not exists(MR):
            makedirs(MR)
        ML = join(originalDst, "MiddleLeft")
        if not exists(ML):
            makedirs(ML)
        MC = join(originalDst, "MiddleCenter")
        if not exists(MC):
            makedirs(MC)
        UD = join(originalDst, "Undefined")
        if mainCounter is None:
            mainCounter = 1
        margin1 = 10.0
        margin2 = -10.0
        margin3 = 10.0
        margin4 = -10.0
        for (inputs, cp_labels) in tqdm(dataSet):

            inputs = Variable(inputs)
            inputs_cpu = inputs.cpu()
            inputs_cpu = inputs_cpu.detach().numpy()
            num_sample = inputs_cpu.shape[0]
            for idx in range(num_sample):
                img = inputs_cpu[idx] * 255.
                img = img[0, :]
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                HP1 = x[mainCounter-1]['head_pose'][0]
                HP2 = x[mainCounter-1]['head_pose'][1]
                #    HP3 = x[mainCounter-1]['head_pose'][2]
                if HP1 > margin1:
                    if HP2 > margin3:
                        originalDst = BR
                    elif HP2 < margin4:
                        originalDst = BL
                    elif margin4 <= HP2 <= margin3:
                        originalDst = BC

                elif HP1 < margin2:
                    if HP2 > margin3:
                        originalDst = TR
                    elif HP2 < margin4:
                        originalDst = TL
                    elif margin4 <= HP2 <= margin3:
                        originalDst = TC

                elif margin2 <= HP1 <= margin1:
                    if HP2 > margin3:
                        originalDst = MR
                    elif HP2 < margin4:
                        originalDst = ML
                    elif margin4 <= HP2 <= margin3:
                        originalDst = MC
                else:
                    originalDst = UD

                file_name = join(originalDst, prefix + str(mainCounter) + ".png")
                if not exists(originalDst):
                    os.mkdir(originalDst)
                cv2.imwrite(file_name, img)
                mainCounter += 1

if __name__ == '__main__':
    dst_folder = "./test"
    syth_fold_path = "./iee_imgs"
    kaggle_fold_path = "./kaggledata/training.csv"
    face_detector_path = "./clsdata/mmod_human_face_detector.dat"

    generate_data(syth_fold_path, kaggle_fold_path, face_detector_path, dst_folder)
    