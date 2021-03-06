# -*- coding:utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
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
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def FR_flip(image, label, dsm):
        """
         将图像和label进行左右翻转
        :param image PIL的图像image
        :return: 翻转之后的图像和label
        """
        return image.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT), dsm.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def TB_flip(image,label,dsm):
        """
         将图像和label进行上下翻转
        :param image PIL的图像image
        :return: 翻转之后的图像和label
        """
        return image.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM), dsm.transpose(Image.FLIP_TOP_BOTTOM)


    @staticmethod
    def randomRotation(image, label, dsm, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST), dsm.rotate(random_angle, Image.NEAREST)



    @staticmethod
    def randomColor(image, label, dsm):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label, dsm  # 调整图像锐度



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
