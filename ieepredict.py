"""
created by: jun wang
"""
from imports import torch, np, time, Variable, cv2, tqdm, os, exists, join
from dataSupplier import DataSupplier
import dataSupplier as DS

import dnnModels


def ensure_folder(folder_name):
    if not exists(folder_name):
        os.makedirs(folder_name)
    return


iee_labels = [18, 22, 23, 27, 28, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 69, 43, 44, 45, 46, 47, 48, 70, 49, 52, 55,
              58]



def forward_hook(self, input, output):
    # print("forward hook..")
    self.X = input[0]
    self.Y = output


class IEEPredictor(object):
    def __init__(self, data_path, model_path, alex=False, nClass=0, max_num=0):
        self.data_path = data_path
        self.model_path = model_path
        self.max_num = max_num
        if torch.cuda.is_available():
            weights = torch.load(self.model_path, map_location=torch.device('cuda:0'))
        else:
            weights = torch.load(self.model_path, map_location=torch.device('cpu'))
        if alex:
            self.model = dnnModels.AlexNetIEE(nClass)
            self.model.load_state_dict(weights)
        else:
            self.model = dnnModels.KPNet()
            self.model.load_state_dict(weights.state_dict())
        self.totalInputs = 0
        self.errorInputs = 0

    def load_data(self, data_path):
        data_supplier = DataSupplier(data_path, DS.batch_size, True, DS.pinMemory, self.max_num)
        return data_supplier.get_data_iters()

    def predict(self, data_batches, dst, originalDst, saveFlag, outPutFile, mainCounter, saveImgs, sub):
        idx = 0
        sCounter = mainCounter
        self.all_diff = None
        errorList = list()
        rightbrow = [2, 3]
        leftbrow = [0, 1]
        mouth = [23, 24, 25, 26]
        righteye = [16, 17, 18, 19, 20, 21, 22]
        lefteye = [9, 10, 11, 12, 13, 14, 15]
        noseridge = [4, 5]
        nose = [6, 7, 8]
        if saveFlag:
            outFile = open(outPutFile, 'w')
            outFile.writelines("image,result,avg_error,max_error,worst_component,worst_KP,rightbrow,leftbrow,righteye,"
                               "lefteye,noseridge,nose,mouth,KP0,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP9,KP10,KP11,KP12,"
                               "KP13,KP14,KP15,KP16,KP17,KP18,KP19,KP20,KP21,KP22,KP23,KP24,KP25,KP26,\r\n")
        for (inputs, cp_labels) in tqdm(data_batches):
            self.totalInputs += len(inputs)
            labels = cp_labels["gm"]
            labels_gt = cp_labels["kps"]
            labels_msk = np.ones(labels_gt.numpy().shape)
            labels_msk[labels_gt.numpy() <= 1e-5] = 0

            if torch.cuda.is_available():
                self.model = self.model.cuda()
                inputs = Variable(inputs.cuda())

            else:
                inputs = Variable(inputs)
            if sub is not None:
                self.model.conv2d_1.register_forward_hook(forward_hook)
            predict = self.model(inputs.float())

            # print(self.model.conv2d_1.Y)
            predict_cpu = predict.cpu()
            predict_cpu = predict_cpu.detach().numpy()
            predict_xy1 = DS.transfer_target(predict_cpu, n_points=DS.n_points)
            predict_xy = np.multiply(predict_xy1, labels_msk)
            avg, sum_diff = self.calculate_pixel_distance(labels_gt.numpy(), predict_xy)

            # print(idx, ": INFO: mean pixel error: ", round(avg,2), " pixels")
            worst = []
            wlabel = []
            inputs_cpu = inputs.cpu()
            inputs_cpu = inputs_cpu.detach().numpy()
            num_sample = inputs_cpu.shape[0]
            for idx in range(num_sample):
                img = inputs_cpu[idx] * 255.
                img = img[0, :]
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                if saveImgs:
                    file_name = join(originalDst, str(mainCounter) + ".png")
                    ensure_folder(originalDst)
                    cv2.imwrite(file_name, img)
                max_error = np.max(sum_diff[idx])
                avg_error = np.sum(sum_diff[idx]) / len(sum_diff[idx])
                worst_KP = 0
                label = 0
                worst_label = 0
                worst_component = "None"
                for KP in sum_diff[idx]:
                    if KP > worst_KP:
                        worst_KP = KP
                        worst_label = label
                    label += 1
                wlabel.append(worst_label)
                if rightbrow.count(worst_label) > 0:
                    worst_component = "rightbrow"
                elif leftbrow.count(worst_label) > 0:
                    worst_component = "leftbrow"
                elif righteye.count(worst_label) > 0:
                    worst_component = "righteye"
                elif lefteye.count(worst_label) > 0:
                    worst_component = "lefteye"
                elif nose.count(worst_label) > 0:
                    worst_component = "nose"
                elif noseridge.count(worst_label) > 0:
                    worst_component = "noseridge"
                elif mouth.count(worst_label) > 0:
                    worst_component = "mouth"
                worst.append(worst_component)
                error_KP_list = list()
                label = 0
                rightbrowComponent = "Normal"
                leftbrowComponent = "Normal"
                righteyeComponent = "Normal"
                lefteyeComponent = "Normal"
                noseridgeComponent = "Normal"
                noseComponenet = "Normal"
                mouthComponent = "Normal"
                for KP in sum_diff[idx]:
                    if KP > 4.0:
                        error_KP_list.append("Error")
                        if rightbrow.count(label) > 0:
                            rightbrowComponent = "Error"
                        elif leftbrow.count(label) > 0:
                            leftbrowComponent = "Error"
                        elif righteye.count(label) > 0:
                            righteyeComponent = "Error"
                        elif lefteye.count(label) > 0:
                            lefteyeComponent = "Error"
                        elif nose.count(label) > 0:
                            noseridgeComponent = "Error"
                        elif noseridge.count(label) > 0:
                            noseComponenet = "Error"
                        elif mouth.count(label) > 0:
                            mouthComponent = "Error"
                    else:
                        error_KP_list.append("Normal")
                    label += 1
                if sub is None:
                    if avg_error > 4.0:
                        self.errorInputs += 1
                        #if dst is not None:
                        #    ensure_folder(dst)
                        #    file_name = join(dst, str(mainCounter) + ".png")
                        #    if not exists(file_name):
                        #        cv2.imwrite(file_name, img)
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                elif sub == "mouth":
                    if mouthComponent == "Error":
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                elif sub == "nose":
                    if noseComponenet == "Error":
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                elif sub == "noseridge":
                    if noseridgeComponent == "Error":
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                elif sub == "lefteye":
                    if lefteyeComponent == "Error":
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                elif sub == "righteye":
                    if righteyeComponent == "Error":
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                elif sub == "leftbrow":
                    if leftbrowComponent == "Error":
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                elif sub == "rightbrow":
                    if rightbrowComponent == "Error":
                        outcome = "Wrong"
                    else:
                        outcome = "Correct"
                errorList.append(outcome)
                if saveFlag:
                    outFile.writelines(str(mainCounter) + ".png" + "," + outcome + "," + str(avg_error) + ","
                                       + str(max_error) + "," + worst_component + "," + "KP" + str(worst_label) + "," +
                                       rightbrowComponent + "," + leftbrowComponent + "," + righteyeComponent + "," +
                                       lefteyeComponent + "," + noseridgeComponent + "," + noseComponenet + "," +
                                       mouthComponent + "," + error_KP_list[0] + "," + error_KP_list[1] + "," +
                                       error_KP_list[2] + "," + error_KP_list[3] + "," + error_KP_list[4] + "," +
                                       error_KP_list[5] + "," + error_KP_list[6] + "," + error_KP_list[7] + "," +
                                       error_KP_list[8] + "," + error_KP_list[9] + "," + error_KP_list[10] + "," +
                                       error_KP_list[11] + "," + error_KP_list[12] + "," + error_KP_list[13] + ","
                                       + error_KP_list[14] + "," + error_KP_list[15] + "," + error_KP_list[16] + "," +
                                       error_KP_list[17] + "," + error_KP_list[18] + "," + error_KP_list[19] + "," +
                                       error_KP_list[20] + "," + error_KP_list[21] + "," + error_KP_list[22] + "," +
                                       error_KP_list[23] + "," + error_KP_list[24] + "," + error_KP_list[25] + "," +
                                       error_KP_list[26] + "," + "\r\n")

                mainCounter += 1
            self.save_evidence(idx, inputs, labels_gt, predict_xy, sum_diff, dst, True, sCounter, worst, wlabel)
            sCounter = mainCounter
            idx += 1
        print("Using data in: ", self.data_path, self.totalInputs)
        print("Avg accuracy: ", str(float((self.totalInputs - self.errorInputs) / self.totalInputs) * 100.0) + "%")
        return mainCounter, errorList

    def calculate_pixel_distance(self, coord1, coord2):
        diff = np.square(coord1 - coord2)
        sum_diff = np.sqrt(diff[:, :, 0] + diff[:, :, 1])
        avg = sum_diff.mean()
        if self.all_diff is None:
            self.all_diff = sum_diff
        else:
            self.all_diff = np.concatenate((self.all_diff, sum_diff), axis=0)
        return avg, sum_diff

    def save_evidence(self, b_idx, inputs, labels, predict_xy, sum_diff, dst, gt, mainCounter, worst, wlabel):
        inputs_cpu = inputs.cpu()
        inputs_cpu = inputs_cpu.detach().numpy()
        num_sample = inputs_cpu.shape[0]

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
            return img
        i = 0
        for idx in range(num_sample):
            img = inputs_cpu[idx] * 255.
            img = img[0, :]
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            xy = predict_xy[idx]
            lab_xy = labels[idx]

            #for coidx in range(xy.shape[0]):
            for coidx in wlabel:
                x_p = int(xy[wlabel[i], 0] + 0.5)
                y_p = int(xy[wlabel[i], 1] + 0.5)
                if gt:
                    x_t = int(lab_xy[wlabel[i], 0] + 0.5)
                    y_t = int(lab_xy[wlabel[i], 1] + 0.5)
                    img = update(img, x_p, y_p, x_t, y_t)
                else:
                    img = update(img, x_p, y_p, 0, 0)

            #file_name = join(dst,worst[i],str(mainCounter)+".png")
            file_name = join(dst,str(mainCounter)+".png")
            print(file_name, "max error at: ", iee_labels[np.argmax(sum_diff[idx])], " value: ", np.max(sum_diff[idx]), " mean: ", sum_diff[idx].mean())
            ensure_folder(join(dst,worst[i]))
            cv2.imwrite(file_name,img)
            mainCounter += 1
            i += 1
            # if np.max(sum_diff[idx]) > 4.0:
            # self.errorInputs1 += 1
            # if np.max(sum_diff[idx]) > 8.0:
            # self.errorInputs2 += 1
            # if np.sum(sum_diff[idx])/len(sum_diff[idx]) > 4.0:
            # self.errorInputs3 += 1
            # ensure_folder(dst)
            # cv2.imwrite(file_name,img)
            # if np.sum(sum_diff[idx])/len(sum_diff[idx]) > 8.0:
            # self.errorInputs4 += 1
            # if (np.sum(sum_diff[idx])/len(sum_diff[idx]) > 8.0) or (np.max(sum_diff[idx]) > 4.0):
            # self.errorInputs5 += 1


if __name__ == '__main__':
    # red color is for groundtruth
    dst = "./evidence"
    data_path = DS.iee_train_data
    model_path = DS.best_model_path

    predictor = IEEPredictor(model_path)
    data_batches, _ = predictor.load_data(data_path)
    predictor.predict(data_batches, dst)

