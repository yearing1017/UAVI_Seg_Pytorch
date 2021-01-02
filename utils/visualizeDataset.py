import glob
import numpy as np
import cv2
import random
from skimage import color, exposure


def imageSegmentationGenerator(images_path, segs_path, n_classes):
    # assert python的断言机制，即先判断assert的条件是否满足，不满足返回异常，满足则继续运行
    assert images_path[-1] == '/' 
    assert segs_path[-1] == '/'
    # 遍历文件夹下所有文件
    # glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合
    # sorted()方法来对可迭代的序列排序生成新的序列。此处按照文件名进行字典排序：['image/222_1.png', 'image/222_2.jpg']
    images = sorted(glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob( images_path +"*.jpeg"))
    segmentations = sorted(glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))
    # 随机n_class中颜色，下面代码表示三通道的颜色值
    #colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]
    colors = [(0,0,0),(0,128,128),(128,0,128),(128,0,0),(128,128,128)]
    assert len(images) == len(segmentations) # 判断两个列表的数目是否一样
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    for im_fn, seg_fn in zip(images, segmentations):

        img = cv2.imread(im_fn)#read color pic
        seg = cv2.imread(seg_fn)
        print(np.unique(seg))  # 该句打印出[0 3]代表seg图中有0和3两类

        seg_img = np.zeros_like(seg)

        # seg_img是用来赋色的：colors[c]代表一类的颜色值；colors[c][0]代表该类的0通道的值
        # 下面的三行分别来对每个通道上的各类的颜色值进行赋值，seg只有一个通道，所以才一直是seg[:,:,0]
        for c in range(n_classes):
            seg_img[:, :, 0] += ((seg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg[:, :, 0] == c) * (colors[c][2])).astype('uint8')
        '''
        # cv2读取图片安照BGR模式读取
        eqaimg = color.rgb2hsv(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # 转成HSV图片
        eqaimg[:, :, 2] = exposure.equalize_hist(eqaimg[:, :, 2])
        eqaimg = color.hsv2rgb(eqaimg)
        '''
        #cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        #cv2.imwrite("data/dataset2/i_crop/3vis.png", seg_img)
        print("保存成功！")
        '''
        cv2.imshow(
            "equalize_hist_img",
            cv2.cvtColor((eqaimg * 255.).astype(np.uint8),cv2.COLOR_RGB2BGR))
        '''
        cv2.waitKey()
        

images = "data/dataset1/image_vis/"
annotations = "data/dataset1/anno_vis/"
n_classes = 4

imageSegmentationGenerator(images, annotations, n_classes)
