# -*- coding:utf-8 -*-
"""数据增强
    1.此代码只用来做随机旋转
    2.因为随机性，所以需要对image、label和dsm数据同时操作
"""

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:
    """
    对image、label和dsm数据三者随机旋转
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")


    @staticmethod
    def randomRotation(image, label, dsm_data, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST), dsm_data.rotate(random_angle, Image.NEAREST)


    @staticmethod
    def saveImage(image, path):
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception as e:
        print(str(e))
        return -2


def imageOps(func_name, image, label, dsm_data, img_des_path, label_des_path, dsm_data_des_path, img_file_name, label_file_name, dsm_data_file_name, 
    times=1):
    funcMap = {"randomRotation": DataAugmentation.randomRotation}
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image, new_label, new_dsm = funcMap[func_name](image, label, dsm_data)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + str(_i) + label_file_name))
        DataAugmentation.saveImage(new_dsm, os.path.join(dsm_data_des_path, func_name + str(_i) + dsm_data_file_name))

#opsList = {"randomRotation","randomColor", "randomGaussian"}
opsList = {"randomRotation"}
#opsList = {"FR_flip","TB_flip"}


def threadOPS(img_path, new_img_path, label_path, new_label_path, dsm_data_path, new_dsm_data_path):
    """
    多线程处理事务
    :param src_path: 资源文件
    :param des_path: 目的地文件
    :return:
    """
    # img path
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
    else:
        img_names = [img_path]

    # label path
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
    else:
        label_names = [label_path]

    if os.path.isdir(dsm_data_path):
        dsm_data_names = os.listdir(dsm_data_path)
    else:
        dsm_data_names = [dsm_data_path]

    img_num = 0
    label_num = 0
    dsm_data_num = 0

    # img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1;
    # label num
    for label_name in label_names:
        tmp_label_name = os.path.join(label_path, label_name)
        if os.path.isdir(tmp_label_name):
            print('contain file folder')
            exit()
        else:
            label_num = label_num + 1

    for dsm_data_name in dsm_data_names:
        tmp_dsm_data_name = os.path.join(dsm_data_path, dsm_data_name)
        if os.path.isdir(tmp_dsm_data_name):
            print('contain file folder')
            exit()
        else:
            dsm_data_num = dsm_data_num + 1

    if img_num != label_num != dsm_data_num:
        print('the num of img and label and dsm_data is not equl')
        exit()
    else:
        num = img_num

    for i in range(num):
        img_name = img_names[i]
        print('img:'+ img_name)
        label_name = label_names[i]
        print('label:' + label_name)
        dsm_data_name = dsm_data_names[i]
        print('dsm:' + dsm_data_name)

        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)
        tmp_dsm_data_name = os.path.join(dsm_data_path, dsm_data_name)

        # 读取文件并进行操作
        image = DataAugmentation.openImage(tmp_img_name)
        image.load()
        label = DataAugmentation.openImage(tmp_label_name)
        image.load()
        dsm_data = DataAugmentation.openImage(tmp_dsm_data_name)
        image.load()
        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            threadImage[_index] = threading.Thread(target=imageOps,
                                                   args=(ops_name, image, label, dsm_data, new_img_path, new_label_path, new_dsm_data_path, img_name,
                                                         label_name, dsm_data_name))
            threadImage[_index].start()
            _index += 1
            time.sleep(0.2)


if __name__ == '__main__':
    threadOPS("data/dataset4/road_car_images_292/",
              "data/dataset4/road_car_images_292_aug/",
              "data/dataset4/road_car_anno_292/",
              "data/dataset4/road_car_anno_292_aug/",
              "data/dataset4/dsm_292",
              "data/dataset4/road_car_dsm_292_aug"
              )
