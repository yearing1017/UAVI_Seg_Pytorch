import os

old_937_path = '/Users/yearing1017/矿区地物提取/deeplabv3/data/dataset3/train_images_noaug_3'
images = os.listdir(old_937_path)

new_937_path = '/Users/yearing1017/矿区地物提取/高程数据/dsm_937/'

for image_name in images:
    if  not os.path.exists(new_937_path + image_name):
    	print(image_name)

print('find over!')