import Augmentor

#确定原始图像存储路径以及掩码文件存储路径
p = Augmentor.Pipeline("data/dataset2/road_car_images")
p.ground_truth("data/dataset2/road_car_anno")


#图像旋转：
p.rotate_random_90(probability=0.6)

#图像左右互换： 按照概率0.5执行
p.flip_left_right(probability=0.6)

# 颜色变化
#p.random_color(probability=0.5, min_factor=0.8, max_factor=1.0)

# 对比度
#p.random_contrast(probability = 0.5, min_factor= 0.8, max_factor = 1.0)

#图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
p.zoom(probability=0.8, min_factor=0.8, max_factor=1.6)

# resize 同一尺寸 200 x 200
p. (probability=1,height=320,width=320,resample_filter=u'NEAREST')

#最终扩充的数据样本数
p.sample(1000)