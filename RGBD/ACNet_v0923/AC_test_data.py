import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import PIL.Image
import cv2

# 数据操作
transform = transforms.Compose([
    transforms.ToTensor()
    ])

class TestDataset(Dataset):

    def __init__(self, transform = None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('data/dataset4/test_images'))

    def __getitem__(self, idx):
        image_name = os.listdir('data/dataset4/test_images')[idx]
        image = cv2.imread('data/dataset4/test_images/' + image_name, 1)
        
        dsm_data = cv2.imread('data/dataset4/test_dsm/' + image_name, 0)

        mask = cv2.imread('data/dataset4/test_anno/'+ image_name, 0)
        
        label = torch.tensor(mask, dtype = torch.long)

        if self.transform:
            image = self.transform(image)
            dsm_data = self.transform(dsm_data)

        return image_name, image, dsm_data, label
        

data = TestDataset(transform)
test_dataloader = DataLoader(data, batch_size=1,shuffle=True, num_workers=1)

if __name__ == "__main__":
    for index, (image_name,image, dsm_data, label) in enumerate(test_dataloader):
        print(image.shape)
        print(label.shape)
        if index >1:
            break   
        #print(label)
    
