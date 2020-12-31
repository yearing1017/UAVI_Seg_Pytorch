import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from onehot import onehot
import cv2

# 数据操作
transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class TestDataset(Dataset):

    def __init__(self, transform = None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('data/dataset1/train_images_noaug'))

    def __getitem__(self, idx):
        image_name = os.listdir('data/dataset1/train_images_noaug')[idx]
        #print("image_name:"+"["+str(idx)+"]"+image_name)
        image = cv2.imread('data/dataset1/train_images_noaug/' + image_name, 1)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读label
        #mask_name = os.listdir('data/dataset1/anno_prepped_train')[idx]
        #print("mask_name:"+"["+str(idx)+"]"+mask_name)
        mask = cv2.imread('data/dataset1/train_anno_noaug/'+ image_name, 0)
        #mask = cv2.imread('data/dataset1/anno_prepped_train/'+os.listdir('data/dataset1/anno_prepped_train')[idx],0)
        label = torch.tensor(mask, dtype = torch.long)

        if self.transform:
            image = self.transform(image)

        return image_name, image, label

data = TestDataset(transform)
#test_size = int(len(data))
#no_size = len(data) - train_size
# random_split:按照给定的长度将数据集划分成没有重叠的新数据集组合
#test_dataset, no_dataset = random_split(data, [test_size, no_size])

# 数据加载时会调用 __getitem__内置方法
#train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=4)
test_dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=1)

if __name__ == "__main__":
    for index, (image_name,image, label) in enumerate(test_dataloader):
        print(image.shape)
        print(label.shape)
        if index >1:
            break   
        #print(label)
    
