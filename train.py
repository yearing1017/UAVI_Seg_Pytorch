import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MyData import train_dataloader,test_dataloader
from deeplabv3 import resnet50, ResNet

def train(epoch = 400):
    # 指定第二块gpu
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # 模型建立
    deeplabv3_model = resnet50()
    deeplabv3_model = deeplabv3_model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device) # CrossEntropyLoss适用多分类
    optimizer = optim.SGD(deeplabv3_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_itere_loss = []
    all_test_iter_loss = []

    for epo in range(epoch):
        # 训练部分
        train_loss = 0
        deeplabv3_model.train()
        for index, (image, label) in enumerate(train_dataloader):
            image = image.to(device) #[4,3,640,640]
            label = label.to(device) #[4,4,640,640]

            optimizer.zero_grad()
            output = deeplabv3_model(image)

            loss = criterion(output, label)
            loss.backward()
            iter_loss = loss.item() # 取出数值
            all_train_itere_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            if np.mod(index, 8) == 0:
                line = "epoch {}, {}/{},train loss is {}".format(epo, index, len(train_dataloader), iter_loss)
                print(line)
                # 写到日志文件
                with open('log/logs.txt', 'a') as f :
                    f.write(line)

        # 验证部分
        test_loss = 0
        deeplabv3_model.eval()
        with torch.no_grad():
            for index, (image, label) in enumerate(test_dataloader):
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = deeplabv3_model(image)
                
                loss = criterion(output, label)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

        line_epoch = "epoch train loss = %f, epoch test loss = %f"
                     %(train_loss/len(train_dataloader), test_loss/len(test_dataloader))
        print('line_epoch')
        with open('log/logs.txt', 'a') as f :
            f.write(line_epoch)
        # 每5轮保存一下模型
        if np.mod(epo, 10) == 0:
            torch.save(deeplabv3_model, 'checkpoints/deeplabv3_model_{}.pt'.format(epo))
            print('saveing checkpoints/deeplabv3_model_{}.pt'.format(epo))


if __name__ == "__main__":
    train(epoch=1)
