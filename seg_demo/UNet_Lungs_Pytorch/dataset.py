import pandas as pd 
import numpy as np 
import os
import cv2
import csv
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from onehot import onehot
transform = transforms.Compose([
    transforms.ToTensor()])
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class LungCTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = transform):
        # csv_file:后面创建的包含image和mask的csv
        # skiprows=1代表从第2行开始标注0
        self.image_frame = pd.read_csv(csv_file, skiprows=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.image_frame.ix[idx,0])
        mask_name = os.path.join(self.root_dir,self.image_frame.ix[idx,1])
        # 0表示灰度
        image = cv2.imread(img_name, 0)
        image = cv2.resize(image,(160, 160))
        #image = image.reshape((1, 160, 160))
        if self.transform:
            image = self.transform(image)
        #image = image.transpose(1,0,2)
        #print(image.shape)   
        mask = cv2.imread(mask_name, 0)
        mask = cv2.resize(mask,(160, 160))
        # mask = mask.reshape((1, 160, 160))
        mask = mask/255
        mask = mask.astype('uint8')
        mask = onehot(mask,2)
        mask = mask.transpose(2,0,1)
        mask = torch.FloatTensor(mask)
        #print(mask.shape)
        sample = {'image':image, 'mask':mask}
        return sample

# 准备数据，将image和mask的路径放入新的csv文件
IMAGE_DIR = "Dataset/2d_images/"
MASK_DIR = "Dataset/2d_masks/"
with open('Dataset/Lung_CT_Dataset.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    # 第一行是列名称
    writer.writerow(["filename", "mask"])
    for p in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, p)
        mask_path = os.path.join(MASK_DIR, p)
        writer.writerow([image_path, mask_path])

data = pd.read_csv("Dataset/Lung_CT_Dataset.csv")
# 打乱顺序
data = data.iloc[np.random.permutation(len(data))]
p = int((len(data)*0.9))
train, validation = data[:p], data[p:]
train.to_csv("Dataset/Lung_CT_Train.csv", index=False)
validation.to_csv("Dataset/Lung_CT_Validation.csv", index=False)
# 数据加载
lung_ct_train_dataset = LungCTDataset(csv_file='Dataset/Lung_CT_Train.csv', root_dir='./')
lung_ct_val_dataset = LungCTDataset(csv_file='Dataset/Lung_CT_Validation.csv', root_dir='./')
train_dataloader = DataLoader(lung_ct_train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(lung_ct_val_dataset, batch_size=4, shuffle=True, num_workers=4)
