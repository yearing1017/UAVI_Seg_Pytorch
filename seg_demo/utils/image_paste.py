import os
from PIL import Image

width_i = 320
height_i = 320

row_max = 18
line_max = 21

all_path = list()
num = 0
pic_max = line_max * row_max

dir_name = r"/Users/yearing1017/矿区地物提取/predict_test_v0301"

# root文件夹的路径  dirs 路径下的文件夹列表  files路径下的文件列表
for root, dirs, files in os.walk(dir_name):
    files.sort()
    for file in files:
        all_path.append(os.path.join(root,file))

#print(all_path)
# all_path获取每张图片的绝对路径

toImage = Image.new( 'RGB',(width_i*line_max,height_i*row_max))


for i in range(row_max):
    for j in range(line_max):
        # 每次打开图片绝对路路径列表的第一张图片
        pic_fole_head = Image.open(all_path[num])
        # 计算每个图片的左上角的坐标点(0, 0)，(0, 320)，(0, 640)
        #loc = (int(i % line_max * width_i), int(j % line_max * height_i))
        loc = (int(j % line_max * height_i), int(i % line_max * width_i))
        print("第{}张图的存放位置".format(num),loc)
        
        toImage.paste(pic_fole_head, loc)
        print(all_path[num])
        num = num + 1

        if num >= len(all_path):
            print("breadk")
            break
    if num >= pic_max:
        break

print(toImage.size)
toImage.save('merged-0301.png')
