from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from dataset import LungCTDataset, train_dataloader, val_dataloader
from model import *

import os
import csv
import numpy as np
import pandas as pd

def train(epoch):
    vis = visdom.Visdom()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型
    model_instance = UNet(1,2)
    model_instance = model_instance.to(device)
    # 优化器和损失函数
    optimizer = optim.Adam(model_instance.parameters(), lr=0.000001)
    criterion = nn.BCELoss().to(device)
    all_train_iter_loss = []
    all_test_iter_loss  = []
    # correct = 0
    for epo in range(epoch):
        train_loss=0
        model_instance.train()
        for batch_idx, data in enumerate(train_dataloader):
            data, target = Variable(data["image"]), Variable(data["mask"])
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model_instance(data)
            loss = criterion(output, target)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            # 以下4行为了取出真正的图片
            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 1, 512, 512)
            #print("output_np.shape")
            #print(output_np.shape)
            output_np = np.argmin(output_np, axis=1)
            target_np = target.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 1, 512, 512) 
            #print("target_np.shape")
            #print(target_np.shape)
            target_np = np.argmin(target_np, axis=1)
            if np.mod(batch_idx, 10) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, batch_idx, len(train_dataloader), iter_loss))
                
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                vis.images(target_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

        test_loss = 0
        model_instance.eval()
        with torch.no_grad():
                    
            for batch_idx, data in enumerate(val_dataloader):
                data, target = Variable(data["image"]), Variable(data["mask"])
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model_instance(data)
                loss = criterion(output, target)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                # 以下4行为了取出真正的图片
                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 1, 512, 512)  
                output_np = np.argmin(output_np, axis=1)
                target_np = target.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 1, 512, 512) 
                target_np = np.argmin(target_np, axis=1)
                if np.mod(batch_idx, 10) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                    vis.images(target_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss',opts=dict(title='test iter loss'))
                
        print('epoch train loss = %f, epoch test loss = %f'
                %(train_loss/len(train_dataloader), test_loss/len(val_dataloader)))

        # 每5轮保存一下模型
        if np.mod(epo, 5) == 0:
            torch.save(model_instance, 'checkpoints/unet_model_{}.pt'.format(epo))
            print('saveing checkpoints/unet_model_{}.pt'.format(epo))

if __name__ == '__main__':
    train(epoch=15)
