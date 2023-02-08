# DL_Stu_Demo
⏰**记录自己做过的一些DL实验**
## 💡 DenseNet_CIFAR-10
- 该项目完整文件如上
- 本实验测试了DenseNet网络模型在CIFAR-10数据集上的表现
- cifar10.py为数据处理文件
- Densenet_Cifar10.py为主程序，运行即可
- 未解决小bug：cifar10.py文件中，在解压部分，可能会报”不是gzip文件“，可手动解压即可
- 博文地址：[DenseNet_CIFAR-10](http://yearing1017.cn/2019/11/28/DenseNet-CIFAR-10/)

## 💡 ResNet50_CIFAR-10
- 在学习了ResNet的基本结构和思想之后，使用**ResNet50+TensorFlow+CIAFR-10**来深入学习网络结构。
- 该项目所使用的环境如下：
  - **Python3.6+TensorFlow1.13+numpy1.16**
  - **CUDA 8.061 + CUDNN 5.1 +tensorflow_gpu 1.2.0**
- 博文地址：[CIFAR-10_ResNet50](http://yearing1017.cn/2019/09/30/CIFAR10-ResNet50/)
- 项目简介：
  - ResNet_model.py：建立ResNet50网络的结构，返回经过该网络训练后的结果
  - CIFARHelper.py：对CIFAR10数据集进行一些特定的处理，及加载数据等功能
  - Main.py：主函数，结合上述文件，使用残差网络对数据集进行训练

## 💡 Tensorflow_Discuz_验证码识别
使用深度学习框架Tensorflow训练出一个用于破解Discuz验证码的模型
- 环境：**GTX1080的显卡，GPU，安装tensorflow_gpu版本，CUDA，CUDNN**。
- 运行train.py即可训练完模型
- 关于本项目的具体博文地址：[Tensorflow_Discuz](http://yearing1017.cn/2019/10/23/Tensorflow-Discuz%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB/)
- Discuz验证码数据集下载地址：
  - [链接](https://pan.baidu.com/s/10TzEvjToYjOgzlOpIgx1tg&shfl=sharepset)  密码:6qto
  
## 💡 Kaggle_Cat&Dog
- 本项目是kaggle中的入门级项目，本篇博文采用**Keras+CNN**网络模型完成该项目。
- kaggle的项目地址：[链接](https://www.kaggle.com/c/dogs-vs-cats/overview)
- kaggle_NoteBook: [notebook地址](https://www.kaggle.com/yearing1017/keras-cnn)
- 详细博文地址：[kaggle_Cat&Dog](http://yearing1017.cn/2019/11/18/Kaggle-Cat-Dog/)

## 💡 Kaggle_Cat&Dog_TransferLearning
- 本项目是kaggle中的入门级项目，本实验采用**Keras+InceptionV3**网络模型完成该项目。
- 本实验采用了迁移学习的思想，借助已有的网络模型，不进行训练而直接进行识别任务。
- kaggle的项目地址：[链接](https://www.kaggle.com/c/dogs-vs-cats/overview)
- kaggle_NoteBook: [notebook地址](https://www.kaggle.com/yearing1017/dogs-vs-cats-inceptionv3-fine-tuning)
- 详细博文地址：[kaggle_Cat&Dog_迁移学习](http://yearing1017.cn/2019/11/19/Kaggle-Cat-Dog-%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0/)

## 💡 Pytorch_FCN
- 一个语义分割（基于FCN）的简单demo
- pytorch框架实现，具体所需环境如下
  - CUDA 9.x （可选）
  - numpy、os、datetime、matplotlib
  - pytorch 1.0
  - torchvision
  - visdom  可视化工具
  - OpenCV-Python
- 具体环境搭建步骤见[博文](http://yearing1017.cn/2019/12/03/FCN-Pytorch/)
- 本实验分为数据处理、网络搭建、训练部分
- 数据集分为data和data_mask部分，即一个为原始数据，一个为标注好的ground_truth
- 文件全都在上面Pytorch_FCN文件夹中

## 💡 Pytorch_UNet
- 将FCN实验网络结构改为UNet进行测试效果
- 搭建了Unet网络，并进行了层次的封装
- 代码：[UNet.py](https://github.com/yearing1017/DL_Stu_Demo/blob/master/Pytorch_FCN/UNet.py)、[train_unet.py](https://github.com/yearing1017/DL_Stu_Demo/blob/master/Pytorch_FCN/train_unet.py)
- 修改train.py,解决了没有预料到的小错误
- 错误1：详见[Pytorch_已解决问题_1](https://github.com/yearing1017/PyTorch_Note)
- 错误2：在执行`loss = criterion(output, bag_msk)`语句时，提示两者大小不同，output比预设的label小
  - 发现是因为在unet的卷积操作中没有添加padding，令padding=1，打印一下，两者相同。程序可运行
- 在执行20轮之后，与FCN的效果进行比较
- 下图为FCN的打印loss信息和可视化结果：

<img src="https://github.com/yearing1017/DL_Stu_Demo/blob/master/Pytorch_FCN/result/fcn_res20.jpg"  style="zoom: 50%;" />

<img src="https://github.com/yearing1017/DL_Stu_Demo/blob/master/Pytorch_FCN/result/fcn_20.jpg" style="zoom: 50%" />


- 下图为UNet的20轮之后的loss信息和可视化结果：

![unet_res20](https://github.com/yearing1017/DL_Stu_Demo/blob/master/Pytorch_FCN/result/unet_res.jpg)

![unet20](https://github.com/yearing1017/DL_Stu_Demo/blob/master/Pytorch_FCN/result/unet_20.jpg)

- 对比两次实验结果，发现UNet的训练损失减少了3个百分点，测试损失减少了5个百分点，在可视化结果上，UNet表现更好

## 💡 UNet_Lungs_Pytorch
- 项目地址：[UNet_Lungs_Pytorch](https://github.com/yearing1017/DL_Stu_Demo/tree/master/UNet_Lungs_Pytorch)
- 本次实验数据来自Kaggle的竞赛题目：[Finding and Measuring Lungs in CT Data](https://www.kaggle.com/kmader/finding-lungs-in-ct-data)
- 实验只用到了2D的数据，所以仅下载2D的数据即可，解压至项目根目录/Dataset/
- dataset.py,onehot.py：数据处理部分
- model.py：UNet模型搭建部分
- train.py：训练部分，自行修改epoch数目，本次仅训练15次
- 可视化结果如下：
<img src="https://github.com/yearing1017/DL_Stu_Demo/blob/master/UNet_Lungs_Pytorch/epoch15.png" style="zoom:50%"/>
