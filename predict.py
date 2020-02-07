import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from onehot import onehot
import cv2
import numpy as np
from test_data import test_dataloader

colors = [(0,0,0),(0,128,128),(128,0,128),(128,0,0)]

def label2color(colors, n_classes, predict):
    for i in range(4):
        seg_color = np.zeros((4, predict.shape[1], predict.shape[2], 3))
        for c in range(n_classes):
            seg_color[i,:, :, 0] += ((predict[i,:,:] == c) *
                                (colors[c][0])).astype('uint8')
            seg_color[i,:, :, 1] += ((predict[i,:,:] == c) *
                                (colors[c][1])).astype('uint8')
            seg_color[i,:, :, 2] += ((predict[i,:,:] == c) *
                                (colors[c][2])).astype('uint8')
        seg_color = seg_color.astype(np.uint8)
    return seg_color

def save(save_dir, image_name_batch, seg_color):
    for idx, image_name in enumerate(image_name_batch):
        seg = seg_color[idx,:,:,:]
        cv2.imwrite(save_dir+image_name, seg)
        print(image_name+"已保存!")

def predict():

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = torch.load('checkpoints_152/deeplabv3_model_200.pt')
    save_dir = 'predict_train_noaug_200_152/'
    model.eval()
    with torch.no_grad():
        for index, (image_name_batch, image, label) in enumerate(test_dataloader):
            image = image.to(device)
            #label = label.to(device)
            predict = model(image) #(4,5,640,640)
            predict_index = torch.argmax(predict, dim=1, keepdim=False).cpu().numpy()  #(4, 640,640)
            seg_color = label2color(colors, 4, predict_index)
            save(save_dir, image_name_batch, seg_color)

if __name__ == "__main__":
    predict()

