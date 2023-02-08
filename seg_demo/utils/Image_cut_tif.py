#!/D:/PY/python
# -*- coding:utf-8 -*-
# author: xiaoyu

import os
import numpy
from osgeo import gdal

class GRID:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件

        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

        del dataset
        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

            # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset

if __name__ == "__main__":
    os.chdir(r'D:\\DEM\\wo_zi_ji_de_test\\2019.11.30')  # 切换路径到待处理图像所在文件夹
    proj, geotrans, data = GRID().read_img('4D_Data.tif')  # 读数据
    print(proj)
    print(geotrans)
    # print(data)
    print(data.shape)
    channel, width, height = data.shape
    for i in range(width // 640):  # 切割成640*640小图
        for j in range(height // 640):
            cur_image = data[:, i * 640:(i + 1) * 640, j * 640:(j + 1) * 640]
            GRID().write_img(r'D:\\DEM\\wo_zi_ji_de_test\\Dataset\\tif_cut_%d_%d.tif' % (i, j), proj, geotrans,
                             cur_image)  ##写数据
