import os
import shutil

# 之前筛选的937图像路径
old_937_path = '/Users/yearing1017/矿区地物提取/deeplabv3/data/dataset3/train_images_noaug_3'
images = os.listdir(old_937_path)

#print('old_937_path len:' + str(len(images)))

# 新的1890高程数据文件夹
new_1890_path = '/Users/yearing1017/矿区地物提取/高程数据/dsm_1890/'

# 目标文件夹，复制1890的文件到此新的dsm_937文件夹
new_937_path = '/Users/yearing1017/矿区地物提取/高程数据/dsm_937/'

# 遍历新的1890高程数据文件夹
for image_name in images:
    if  os.path.exists(new_1890_path + image_name):
        #print(image_name)
        old_name = new_1890_path + image_name
        new_name = new_937_path + image_name
        shutil.copyfile(old_name,new_name)
        print('已复制:' + image_name)
print('old_937_path len:' + str(len(images)))
print("over")
