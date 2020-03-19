#  Deeplabv3_Pytorch
- 基于矿区无人机影像的地物提取实验
## 1. 实验数据介绍
- 一副无人机拍摄的高分辨率矿区影像图
- 实验室进行标注的对应label
- v0219版本：进行裁剪后的640 x 640的图像与label数据
- v0225&v0301版本：进行裁剪后的640 x 640的图像与label数据，并更换测试集

## 2. 实验环境介绍
- GPU等服务器资源不加介绍
- Python3.6、Pytorch、OpenCV、torchvision、numpy等必备环境
- 图像切割工具包GDAL：仅在win系统下可运行
## 3. 实验流程介绍
- 原图数据和标注好的label数据，label是灰度的图像，且每个像素属于该类（0-3）共四类
- 切割原图和相应的label数据为640 x 640的图像
- 将切割好的原图和label对应好，使用代码进行可视化（因为标注的label是灰度，直观上看是黑色一片）
- 对数据进行数据增强，切割完成才有对应的几百张数据，增强至几千张
- 增强后的数据也要一对一对应好，建议一开始就取两者相同的图像名，使用可视化进行测试
- 数据划分，分为训练集、验证集、测试集；按照6:2:2的比例
- 搭建网络代码，使用Pytorch搭建deeplabv3网络（基于ResNet）
- 编写train.py训练代码，写好训练流程、保存模型、保存loss等信息
- 训练完成之后，使用保存的模型进行预测,对预测出的图片进行涂色，使之可视化
- 根据预测结果进行kappa系数、mIoU指标的计算
## 4. 实验详细流程
- [数据简介](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/%E6%95%B0%E6%8D%AE%E7%AE%80%E4%BB%8B.md)
- [数据切割](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/%E6%95%B0%E6%8D%AE%E5%88%87%E5%89%B2.md)
- [灰度label可视化](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/%E7%81%B0%E5%BA%A6label%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- [数据增强_Data_Augmentation.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/Data_Augmentation.py)
- [数据载入及数据划分](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MyData.py)
- [deeplabv3.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/deeplabv3.py)：基于pytorch的deeplab_v3网络搭建代码
- [deeplabv3论文笔记](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/Deeplab_v3.md)
- [train.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/train.py)：训练程序的代码
- [predictv0217.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/predictv0217.py)：使用生成的训练模型来预测，并给预测图片进行涂色显示
- [MIoU指标计算](http://yearing1017.cn/2020/02/17/MIoU-PyTorch/)
- [Kappa系数计算](http://yearing1017.cn/2020/02/27/基于混淆矩阵的Kappa系数的计算/)

## 5. 实验数据记录
- 基于ResNet-152的deeplabv3训练epoch=200得到的具体参数及有关结果如下：
  - 使用图像增强进行训练集的生成，640大小的图像：训练集1944张、验证集648张、测试集162张
  - v0225版本&v0301版本，320大小的图像：训练集9072张、验证集2268张、测试集378张
  - 图像的预处理：无标准化、只进行了归一化
  - 损失函数：CrossEntropyLoss
  - 优化器：SGD、lr=1e-3、动量=0.9；  v0219版本测试Adam优化器、lr=1e-3
  - v0225版本&v0301版本:Adam优化器、lr=1e-3
  - v0225版本：deeplabv3-resnet152
  - v0301版本：deeplabv3-resnet50
  - v0304版本：deeplabv3-resnet152，将简单留出验证改为5折交叉验证
  - V3p-101： deeplabv3+-resnet101 + 5折交叉验证
  - V3p-152： deeplabv3+-resnet152 + 5折交叉验证

### 5.1 test测试数据集-v0219

#### 5.1.1 SGD与Adam整体平均像素acc及miou对比-下为Adam

|         acc        |        MIoU        |
| :----------------: | :----------------: |
| 0.9263767325704164 | 0.4807448577750288 |
| 0.9337385405707201 | 0.47286513489126114|

#### 5.1.2 类别平均像素acc-v0219

| 类别 |        SGD         |        Adam        |
| :--: | :----------------: | :----------------: |
|  地面   | 0.9280041488314654 | 0.9393157770328366 |
|  房屋   | 0.8031034186590591 | 0.8322606620969475 |
|  道路   | 0.522966884580534  | 0.7378283121400184 |
|  车辆   | 0.6060759535374916 | 0.7527768185633605 |

### 5.2 test测试数据集-各版本结果对比

|     版本&指标    |  Acc   |  MIoU  | Kappa  |  地面  |  房屋  |  道路  |  车辆  |
| :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|  V0225  | 0.9562 | 0.7864 | 0.8477 | 0.9847 | 0.8531 | 0.8898 | 0.7269 |
|  V0301  | 0.9545 | 0.7740 | 0.8429 | 0.9815 | 0.8849 | 0.9037 | 0.6313 |
|  V0304  | 0.9555 | 0.7777 | 0.8470 | 0.9816 | 0.8529 | 0.9086 | 0.7151 |
| V3p-101 | 0.9544 | 0.7706 | 0.8427 | 0.9785 | 0.8704 | 0.9165 | 0.5758 |
| V3p-152 | 0.9525 | 0.7582 | 0.8317 | 0.9860 | 0.8235 | 0.9111 | 0.6009 |

## 6. 实验分割结果展示
- 整体效果不行，如下，v0217版本已解决,详见下方优化结果：
![](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/image/duibi1.jpg)

- 单张的效果还可以，如下：
![](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/image/predict.png)

## 7. 实验待优化问题-MIoU-已解决v0217
- MIoU数据：[MIoUData.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MIoU/MIoUData.py)：读取label和predict图像，以tensor形式，batch_size=4传入----v0210
- MIoU的计算：[testMIoU.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MIoU/testMIoU.py)：将传入的tensor转为np的array，再执行flatten()进行一维化，每4个图像进行计算miou，最后求平均的miou
- 问题：计算得到的MIoU都为1.0，怀疑原因，中间的float强转int，最后得到的数值都为1
- 解决v0217：修改读入方式，使用CV2直接读入，不变为tensor，详见[MIoUCalv0217.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MIoU/MIoUCalv0217.py)，[MIoUDatav0217.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/MIoU/MIoUDatav0217.py)
## 8. 实验待优化问题-预处理
- MyData_v0211版本，cv2以BGR模式读入训练集，先改为RGB图像，再进行nomalize初始化，使用的mean和std数值都为Imagenet数据集预训练得到的，但是训练完成之后，预测结果有偏差，如下：
![v0211predict](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/image/1-2.jpg)

## 9. 预测问题-已解决v0217
- v0217版本：修改预测方法，以一张一张读入进行预测，解决之前的大部分涂色失败问题。效果如下：
![](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/image/DUIBI.jpg)

## 10. SGD与Adam优化器预测效果对比
- v0219：仅仅改动优化器为Adam，lr=1e-3
![](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/image/adam-1.jpg)

![](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/image/adam-2.jpg)
