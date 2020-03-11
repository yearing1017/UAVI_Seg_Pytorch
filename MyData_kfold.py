import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split,Subset
from sklearn.model_selection import ShuffleSplit
from torchvision import transforms
from PIL import Image
import cv2

# 数据操作
transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MyDataset(Dataset):

    def __init__(self, transform = None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('data/dataset2/images_prepped_train'))

    def __getitem__(self, idx):
        image_name = os.listdir('data/dataset2/images_prepped_train')[idx]
        #print("image_name:"+"["+str(idx)+"]"+image_name)
        #image = Image.open('data/dataset1/images_prepped_train/' + image_name)
        image = cv2.imread('data/dataset2/images_prepped_train/' + image_name, 1)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读label
        #mask_name = os.listdir('data/dataset1/anno_prepped_train')[idx]
        #print("mask_name:"+"["+str(idx)+"]"+mask_name)
        #label = Image.open('data/dataset1/anno_prepped_train/' + image_name)
        mask = cv2.imread('data/dataset2/anno_prepped_train/'+ image_name, 0)
        #mask = cv2.imread('data/dataset1/anno_prepped_train/'+os.listdir('data/dataset1/anno_prepped_train')[idx],0)
        label = torch.tensor(mask, dtype = torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

data = MyDataset(transform)

# k折交叉验证的实现，n_splits为分为几份，test_size为验证集所占比例
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# 根据k折交叉验证方法分好的索引划分数据集
train_dataset = list()
val_dataset = list()
for train_index, test_index in ss.split(data):
    train_dataset.append(Subset(data, train_index))
    val_dataset.append(Subset(data, test_index))
    #print("%s %s" % (train_index, test_index))


#train_dataset, val_dataset = data[train_index], data[test_index]
#print(train_dataset[4])
#print(len(val_dataset[0]))
# 根据划分生成，每一个加载器有5份数据集
train_dataloader = list()
val_dataloader = list()
for i in range(5):
    train_dataloader.append(DataLoader(train_dataset[i], batch_size=6, shuffle=True,num_workers=6))
    val_dataloader.append(DataLoader(val_dataset[i], batch_size=6,shuffle=True, num_workers=6))

if __name__ == "__main__":
    # 测试 是否 成功载入数据，且train和val的图像没有重复
    for index, (image_name, image, label) in enumerate(val_dataloader[0]):
        print(image.shape)
        print(label.shape)
        print(image_name)
        print('=============')
