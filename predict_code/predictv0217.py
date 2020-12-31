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
    seg_color = np.zeros((predict.shape[1], predict.shape[2], 3))
    for c in range(n_classes):
        seg_color[:, :, 0] += ((predict[0,:,:] == c) *
                            (colors[c][0])).astype('uint8')
        seg_color[:, :, 1] += ((predict[0,:,:] == c) *
                            (colors[c][1])).astype('uint8')
        seg_color[:, :, 2] += ((predict[0,:,:] == c) *
                            (colors[c][2])).astype('uint8')
    seg_color = seg_color.astype(np.uint8)
    return seg_color

def save(save_dir, image_name_batch, seg_color):
    for image_name in image_name_batch:
        seg = seg_color[:,:,:]
        cv2.imwrite(save_dir+image_name, seg)
        print(image_name+"已保存!")

def predict():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = torch.load('checkpoints_152/deeplabv3_model_200.pt')
    save_dir = 'predict_train_noaug_v0217/'
    model.eval()
    with torch.no_grad():
        for index, (image_name_batch, image, label) in enumerate(test_dataloader):
            #print(image_name_batch)
            image = image.to(device)
            #label = label.to(device)
            predict = model(image) #(4,5,640,640)
            predict_index = torch.argmax(predict, dim=1, keepdim=False).cpu().numpy()  #(4, 640,640)
            seg_color = label2color(colors, 4, predict_index)
            save(save_dir, image_name_batch, seg_color)

if __name__ == "__main__":
    predict()

