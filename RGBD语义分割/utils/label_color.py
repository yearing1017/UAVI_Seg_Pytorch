#import glob
import numpy as np
import cv2
import random
from skimage import color, exposure
import os


def imageSegmentationGenerator(segs_path, n_classes):

    colors = [(0,0,0),(0,128,128),(128,0,128),(128,0,0),(128,128,128)]

    #images = os.listdir('data/dataset3/road_car_images')
    anno = os.listdir(segs_path)
    #for index, seg_fn in enumerate(segmentations):
    for anno_name in anno:
        #mg = cv2.imread(im_fn)#read color pic
        seg = cv2.imread(segs_path + anno_name)
        #print(np.unique(seg))  # 该句打印出[0 3]代表seg图中有0和3两类

        seg_img = np.zeros_like(seg)

        # seg_img是用来赋色的：colors[c]代表一类的颜色值；colors[c][0]代表该类的0通道的值
        # 下面的三行分别来对每个通道上的各类的颜色值进行赋值，seg只有一个通道，所以才一直是seg[:,:,0]
        for c in range(n_classes):
            seg_img[:, :, 0] += ((seg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg[:, :, 0] == c) * (colors[c][2])).astype('uint8')
        
       
        #cv2.imwrite("/Users/yearing1017/矿区地物提取/deeplabv3/predict_color_1004/" + anno_name, seg_img)
        cv2.imwrite("/Users/yearing1017/矿区地物提取/deeplabv3/predict_color_1008/" + anno_name, seg_img)
        print(anno_name+"：保存成功！")


#images = "data/dataset1/image_vis/"
#annotations = "/Users/yearing1017/矿区地物提取/deeplabv3/predict_gray_1004/"
annotations = "/Users/yearing1017/矿区地物提取/deeplabv3/predict_gray_1008/"
n_classes = 4

imageSegmentationGenerator(annotations, n_classes)
