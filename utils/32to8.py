import cv2
import numpy as np
from skimage import io

'''
# 读取tif为ndarray,测试最小值和最大值
img = io.imread('souhtcon_dsm.tif')
min_value = 1165.91
max_value = 1217.37
for i in range(27497):
	for j in range(31541):
		if img[i][j]>0 and img[i][j]<min_value:
			min_value = img[i][j]
		elif img[i][j] > max_value:
			max_value = img[i][j]
print('查找结束,min和max如下：')
print(min_value) # 1165.8558
print(max_value) # 1217.3722
'''

# 读取各像素减去最小值，转为8位；并保存8位的文件
img = io.imread('souhtcon_dsm.tif')
value = 1165.8558
print('开始读取修改：')
for i in range(27497):
	for j in range(31541):
		if img[i][j]<0:
			img[i][j] = 0.0
		else:
			img[i][j] -= value
			img[i][j] = int(img[i][j])
print('修改完成！')
print('开始保存')
io.imsave('souhtcon_dsm_code8.tif', img)


# 转换代码如下
img = io.imread('souhtcon_dsm_code8.tif')
img_f_8 = img.astype(dtype = np.uint8)
io.imsave('souhtcon_dsm_code&8.tif', img_f_8)
print('end')
