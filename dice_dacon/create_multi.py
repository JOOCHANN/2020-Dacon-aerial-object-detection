from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections, return_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
import numpy as np
from tqdm import tqdm
import DOTA_devkit.polyiou as polyiou
import math
import pdb
from os import listdir
import csv
import pandas as pd
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--testdir_path', help='the dir to save logs and models',default= "/data/hdd/Dacon/dota/test/images")
    args = parser.parse_args()
    return args
def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class DetectorModel():
    def __init__(self,
                 config_file,
                 checkpoint_file):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        self.classnames = self.dataset.CLASSES
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        # self.cnt =0
    def inference_single(self, imagname, slide_size, chip_size):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        slide_h, slide_w = slide_size
        hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]

        for i in range(int(width / slide_w + 1)):
            for j in range(int(height / slide_h) + 1):
                subimg = np.zeros((hn, wn, channel))
                # print('i: ', i, 'j: ', j)
                chip = img[j*slide_h:j*slide_h + hn, i*slide_w:i*slide_w + wn, :3]
                subimg[:chip.shape[0], :chip.shape[1], :] = chip

                chip_detections = inference_detector(self.model, subimg)

                # print('result: ', result)
                for cls_id, name in enumerate(self.classnames):
                    chip_detections[cls_id][:, :8][:, ::2] = chip_detections[cls_id][:, :8][:, ::2] + i * slide_w
                    chip_detections[cls_id][:, :8][:, 1::2] = chip_detections[cls_id][:, :8][:, 1::2] + j * slide_h
                    # import pdb;pdb.set_trace()
                    try:
                        total_detections[cls_id] = np.concatenate((total_detections[cls_id], chip_detections[cls_id]))
                    except:
                        import pdb; pdb.set_trace()
        # nms
        for i in range(len(self.classnames)):
            keep = py_cpu_nms_poly_fast_np(total_detections[i], 0.1)
            total_detections[i] = total_detections[i][keep]
        return total_detections

    def inference_single_csv(self, srcpath):
        detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]
        detections_1024 = self.inference_single(srcpath, (512, 512), (1024, 1024))
        detections_1500 = self.inference_single(srcpath, (750, 750), (1500, 1500))
        detections_3000 = self.inference_single(srcpath, (3000, 3000), (3000, 3000))
 
        detections[0] = np.concatenate((detections_1024[0],detections_1500[0],detections_3000[0]))
        detections[1] = np.concatenate((detections_1024[1], detections_1500[1], detections_3000[1]))
        detections[2] = np.concatenate((detections_1024[2], detections_1500[2], detections_3000[2]))
        detections[3] = np.concatenate((detections_1024[3], detections_1500[3], detections_3000[3]))

        # nms
        for i in range(len(self.classnames)):
            keep = py_cpu_nms_poly_fast_np(detections[i], 0.1)
            detections[i] = detections[i][keep]
        result = return_poly_detections(srcpath, detections, self.classnames, scale=1, threshold=0.01)

        # img = draw_poly_detections(srcpath, detections, self.classnames, scale=1, threshold=0.01)
        # img_name = '/data/hdd/Dacon/monitoring/'+str(self.cnt)+'.png'
        # cv2.imwrite(img_name,img)
        # self.cnt += 1
        return result


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

if __name__ == '__main__':
    args = parse_args()
    roitransformer = DetectorModel(r'configs/DOTA/faster_rcnn_RoITrans_resnetx101_bfn_1x_dota_scale.py',
                  r'work_dirs/faster_rcnn_RoITrans_resnetx101_bfn_1x_dota_aug_scale_v2/epoch_13.pth')

    #test imag path
    image_dir = args.testdir_path
    image_filenames = [os.path.join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
    result = [['file_name','class_id','confidence','point1_x','point1_y','point2_x','point2_y','point3_x','point3_y','point4_x','point4_y']]

    for i in range(len(image_filenames)):
        tmp = roitransformer.inference_single_csv(image_filenames[i])
        result = result + tmp
        if (i%20 == 0) :
            print(i, '/', len(image_filenames))

    print("number of row :", len(result))
    dataframe = pd.DataFrame(result)
    
    #name of csv file
    dataframe.to_csv("faster_rcnn_RoITrans_resnetx101_bfn_1x_dota_scale_uncertain_epoch13.csv",header=False,index=False)  
