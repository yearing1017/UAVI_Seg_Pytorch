# -*- coding:utf-8 -*-
"""dsm高程数据的单独增强
   1. 单独翻转变换dsm数据 flip
   2. 颜色抖动：返回源数据，为了匹配image和label
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
    def FR_flip(image):
        """
         将图像和label进行左右翻转
        :param image PIL的图像image
        :return: 翻转之后的图像和label
        """
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def TB_flip(image):
        """
         将图像和label进行上下翻转
        :param image PIL的图像image
        :return: 翻转之后的图像和label
        """
        return image.transpose(Image.FLIP_TOP_BOTTOM)


    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        return image


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


def imageOps(func_name, image, img_des_path, img_file_name, times=2):
    funcMap = {"randomColor": DataAugmentation.randomColor,
               "FR_flip":DataAugmentation.FR_flip,
               "TB_flip":DataAugmentation.TB_flip
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image = funcMap[func_name](image)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))


#opsList = {"randomRotation","randomColor", "randomGaussian"}
#opsList = {"randomRotation"}
#opsList = {"FR_flip","TB_flip"}
opsList = {"randomColor"}


def threadOPS(img_path, new_img_path):
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

    img_num = 0

    # img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1;

    num = img_num

    for i in range(num):
        img_name = img_names[i]
        print(img_name)

        tmp_img_name = os.path.join(img_path, img_name)


        # 读取文件并进行操作
        image = DataAugmentation.openImage(tmp_img_name)
        image.load()

        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            threadImage[_index] = threading.Thread(target=imageOps,
                                                   args=(ops_name, image, new_img_path, img_name))
            threadImage[_index].start()
            _index += 1
            time.sleep(0.2)


if __name__ == '__main__':
    threadOPS("/Users/yearing1017/矿区地物提取/deeplabv3/data/dataset4/dsm_937",
              "/Users/yearing1017/矿区地物提取/deeplabv3/data/dataset4/dsm_937_color/")
