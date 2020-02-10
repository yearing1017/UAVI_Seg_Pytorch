#  Deeplabv3_Pytorch
- 基于矿区无人机影像的地物提取实验
## 实验数据介绍
- 一副无人机拍摄的高分辨率矿区影像图
- 实验室进行标注的对应label
- 进行裁剪后的640 x 640的图像与label数据
## 实验环境介绍
- GPU等服务器资源不加介绍
- Python3.6、Pytorch、OpenCV、torchvision、numpy等必备环境
- 图像切割工具包GDAL：仅在win系统下可运行
## 实验流程介绍
- 原图数据和标注好的label数据，label是灰度的图像，且每个像素属于该类（0-3）共四类
- 切割原图和相应的label数据为640 x 640的图像
- 将切割好的原图和label对应好，使用代码进行可视化（因为标注的label是灰度，直观上看是黑色一片）
- 对数据进行数据增强，切割完成才有对应的几百张数据，增强至几千张
- 增强后的数据也要一对一对应好，建议一开始就取两者相同的图像名，使用可视化进行测试
- 数据划分，分为训练集、验证集、测试集；按照6:2:2的比例
- 搭建网络代码，使用Pytorch搭建deeplabv3网络（基于ResNet）
- 编写train.py训练代码，写好训练流程、保存模型、保存loss等信息
- 训练完成之后，使用保存的模型进行预测,对预测出的图片进行涂色，使之可视化
## 实验详细流程
- [数据简介](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/%E6%95%B0%E6%8D%AE%E7%AE%80%E4%BB%8B.md)
- [数据切割](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/%E6%95%B0%E6%8D%AE%E5%88%87%E5%89%B2.md)
- [灰度label可视化](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/%E7%81%B0%E5%BA%A6label%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- [数据增强_Data_Augmentation.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/Data_Augmentation.py)
- [数据载入及数据划分](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MyData.py)
- [deeplabv3.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/deeplabv3.py)：基于pytorch的deeplab_v3网络搭建代码
- [deeplabv3论文笔记](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/Deeplab_v3.md)
- [train.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/train.py)：训练程序的代码
- [predict.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/predict.py)：使用生成的训练模型来预测，并给预测图片进行涂色显示

## 实验分割结果展示
![](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/image/predict.png)

## 实验待优化问题
- MIoU数据：[MIoUData.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MIoU/MIoUData.py)：读取label和predict图像，以tensor形式，batch_size=4传入----v0210
- MIoU的计算：[testMIoU.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MIoU/testMIoU.py)：将传入的tensor转为np的array，再执行flatten()进行一维化，每4个图像进行计算miou，最后求平均的miou

