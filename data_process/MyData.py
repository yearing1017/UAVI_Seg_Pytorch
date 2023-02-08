import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from onehot import onehot
import cv2

# 数据操作
transform = transforms.Compose([
    transforms.ToTensor(),
    # 猜测由于该参数+RGB转换导致预测混乱
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
])

class MyDataset(Dataset):

    def __init__(self, transform = None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('data/dataset1/images_prepped_train'))

    def __getitem__(self, idx):
        image_name = os.listdir('data/dataset1/images_prepped_train')[idx]
        image = cv2.imread('data/dataset1/images_prepped_train/' + image_name)
        # 猜测由于下句BGR-->RGB转换导致预测混乱
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读label,因为对应的label和原图对应，取一致的名字
        #mask_name = os.listdir('data/dataset1/annotations_prepped_train')[idx]
        mask = cv2.imread('data/dataset1/annotations_prepped_train/'+ image_name, 0)
        label = torch.LongTensor(mask)

        if self.transform:
            image = self.transform(image)

        return image, label

data = MyDataset(transform)
train_size = int(0.75 * len(data))
test_size = len(data) - train_size
# random_split:按照给定的长度将数据集划分成没有重叠的新数据集组合
train_dataset, test_dataset = random_split(data, [train_size, test_size])

# 数据加载时会调用 __getitem__内置方法
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ == "__main__":
    for index, (image, label) in enumerate(train_dataloader):
        print(image.shape)
        print(label.shape)
    
