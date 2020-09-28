## RGBD语义分割实验总结

### 1. [ACNet_v0923-v0925版本](https://github.com/yearing1017/Deeplabv3_Pytorch/tree/master/RGBD%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/ACNet_v0923)

#### 1.1 实验简介
- 网络：ACNet原始代码，未改动；
- ACNet_v0923baseline：resnet50的layer1-4；
- ACNet_v0925baseline：resnet101的layer1-4；
- 训练与验证：5折交叉验证；
- 训练轮次数：epoch = 40；
- 模型的评价标准：MIoU；

#### 1.2 实验数据
- 数据：7830张320 x 320尺寸的image、dsm_data、label；
- **数据处理部分**
  - Depth原始数据（高程数据）是32位的tiff格式图像，需转换为8位的tiff格式图，**使用python的代码直接转换会线性压缩图像的对比性，所以要先找到图像像素的极值，将其像素之间的差做出来，再使用代码转换为uint8文件**。转换代码地址：[32to8.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/utils/32to8.py)
  - png切割代码：[png_crop.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/utils/png_crop.py)
  - 在dataset3中有筛选出的特征较均衡的图像，使用代码筛选切割好的dsm高程数据；代码：[file_find.py](https://github.com/yearing1017/Deeplabv3_Pytorch/blob/master/utils/file_find.py)
  - 基于dataset3加入dsm高程数据：[dataset4的调整记录](https://github.com/yearing1017/Deeplabv3_Pytorch/issues/2#issuecomment-696058635)

#### 1.3 实验结果
- 较[ccnet_v3_0607](https://github.com/yearing1017/CCNet_PyTorch)版本各项指标均有提升；
- 具体acc及miou指标如下：

|     版本&指标    |  Acc   |  MIoU  | Kappa  |  背景  |  房屋  |  道路  |  车辆  |
| :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| CCNet0607 | 0.9603 | 0.8057 | 0.8684 | 0.9722 | 0.8907 | 0.9216 | 0.7745 |
| ACNet0923 | 0.9635 | 0.8220 | 0.8802 | 0.9710 | 0.9191 | 0.9588 | 0.8167 |
| ACNet0925 | 0.9583 | 0.8016 | 0.8625 | 0.9716 | 0.9060 | 0.9002 | 0.7994 |

### 2. [ACNet_v0927版本](https://github.com/yearing1017/Deeplabv3_Pytorch/tree/master/RGBD%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/ACNet_v0927)

#### 2.1 实验简介
- 网络：ACNet的decode阶段普通的卷积层换为CC_Module；
- baseline：resnet50的layer1-4；
- 训练与验证：5折交叉验证；
- 训练轮次数：epoch = 45；
- 模型的评价标准：MIoU；


### 2. [ACNet_v0928版本](https://github.com/yearing1017/Deeplabv3_Pytorch/tree/master/RGBD%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/ACNet_v0928)

#### 2.1 实验简介
- 网络：ACNet的通道注意力换为CC_Module；
- baseline：resnet50的layer1-4；
- 训练与验证：5折交叉验证；
- 训练轮次数：epoch = 45；
- 模型的评价标准：MIoU；
