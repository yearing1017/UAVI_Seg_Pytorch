import os 
import torch
import torch.nn as nn

from torchvision import transforms

from ACNet_model import ACNet

import cv2
import numpy as np
from AC_test_data import test_dataloader

def save(save_dir, image_name_batch, seg_color):
    for idx, image_name in enumerate(image_name_batch):
        seg = seg_color[idx,:,:]
        cv2.imwrite(save_dir+image_name, seg)
        print(image_name+"已保存!")

def predict():

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    model = ACNet()
    model.load_state_dict(torch.load("models_acnet_v0925/acnet_0925_38.pth"))
    model = model.to(device)
    #model = torch.load('checkpoints_v3p_v0316/deeplabv3plus_model_34.pt')
    save_dir = 'predict_gray_0925/'
    
    model.eval()
    with torch.no_grad():
        for index, (image_name_batch, image, dsm_data, label) in enumerate(test_dataloader):
            image = image.to(device)
            dsm_data = dsm_data.to(device)
            predict = model(image, dsm_data) 
            predict_index = torch.argmax(predict, dim=1, keepdim=False).cpu().numpy()  #(4, 640,640)
            #seg_color = label2color(colors, 4, predict_index)
            save(save_dir, image_name_batch, predict_index)

if __name__ == "__main__":
    predict()
