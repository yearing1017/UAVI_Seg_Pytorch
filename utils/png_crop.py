import numpy as np
import matplotlib
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def img_seg(dir):
    files = os.listdir(dir)
    for file in files:
        a, b = os.path.splitext(file)
        img = Image.open(os.path.join(dir + "/" + file))
        hight, width = img.size
        w = 320
        # 切割的命名从0开始
        id = 0
        i = 0  # 横方向
        while (i + w <= width):
            j = 0 # 竖方向      左 上 右 下
            while (j + w <= hight):
                new_img = img.crop((j, i, j + w, i + w))
                # rename = "D:\\labelme\\images\\"
                rename = r'/Users/yearing1017/矿区地物提取/高程数据/dsm_5_320/'
                new_img.save(rename + str(id) + b)
                id += 1
                j += w
            i = i + w

if __name__ == '__main__':
    # path = "D:\\labelme\\data\\images\\train"
    path = r'/Users/yearing1017/矿区地物提取/高程数据/dsm-png-temp/'
    img_seg(path)
