import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import cv2

transform = transforms.Compose([
    transforms.ToTensor()
])


class MIoUDataset(Dataset):

    def __init__(self, transform = None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('data/dataset1/annotations_prepped_train'))

    def __getitem__(self, idx):
        # 读label
        label_name = os.listdir('data/dataset1/annotations_prepped_train')[idx]
        label = Image.open('data/dataset1/annotations_prepped_train/' + label_name)
        # 读 predict
        predict = Image.open('predict/' + label_name)
        if self.transform:
            label = self.transform(label)
            predict = self.transform(predict)
        return label, predict

data = MIoUDataset()

# 数据加载时会调用 __getitem__内置方法
MIoU_dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=1)

if __name__ == "__main__":
    for index, (label, predict) in enumerate(MIoU_dataloader):
        print(label.shape)
        print(predict.shape)
    
