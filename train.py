import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MyData import train_dataloader,val_dataloader
from deeplabv3 import resnet50, resnet101,resnet152, ResNet
from tensorboardX import SummaryWriter
from MIouv0217 import Evaluator



def train(epoch = 400):
    # 创建指标计算对象
    evaluator = Evaluator(4)

    # 定义最好指标miou数值，初始化为0
    best_pred = 0.0
    writer = SummaryWriter('tblog/deeplabv3_v0304')
    # 指定第二块gpu
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # 模型建立
    deeplabv3_model = resnet50()
    #deeplabv3_model = torch.load('checkpoints/deeplabv3_model_90.pt')
    deeplabv3_model = deeplabv3_model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device) # CrossEntropyLoss适用多分类
    optimizer = optim.Adam(deeplabv3_model.parameters(), lr=1e-3)

    #all_train_iter_loss = []
    #all_val_iter_loss = []

    for epo in range(epoch):
        # 训练部分
        train_loss = 0
        deeplabv3_model.train()
        for index, (image, label) in enumerate(train_dataloader):
            image = image.to(device) 
            label = label.to(device)
            optimizer.zero_grad()
            output = deeplabv3_model(image)
            loss = criterion(output, label)
            #print(loss.shape)
            loss.backward()
            iter_loss = loss.item() # 取出数值
            #all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            if np.mod(index, 8) == 0:
                line = "epoch {}, {}/{},train loss is {}".format(epo, index, len(train_dataloader), iter_loss)
                print(line)
                # 写到日志文件
                with open('log/logs_v0304.txt', 'a') as f :
                    f.write(line)
                    f.write('\r\n')

        # 验证部分
        val_loss = 0
        deeplabv3_model.eval()
        with torch.no_grad():
            for index, (image, label) in enumerate(val_dataloader):
                image = image.to(device)
                #label = label.reshape(-1, 5)
                label = label.to(device)

                optimizer.zero_grad()
                output = deeplabv3_model(image)
                loss = criterion(output, label)
                iter_loss = loss.item()
                #all_val_iter_loss.append(iter_loss)
                val_loss += iter_loss
                # 记录相关指标数据
                pred = output.cpu().numpy()
                label = label.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(label, pred)
        
        line_epoch = "epoch train loss = %.3f, epoch val loss = %.3f" % (train_loss/len(train_dataloader), val_loss/len(val_dataloader))
        print(line_epoch)
        with open('log/logs_v0304.txt', 'a') as f :
            f.write(line_epoch)
            f.write('\r\n')
        
        Acc = evaluator.Pixel_Accuracy()
        #Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        # tensorboard记录
        writer.add_scalar('train_loss', train_loss/len(train_dataloader), epo)
        writer.add_scalar('val_loss', val_loss/len(val_dataloader), epo)
        writer.add_scalar('Acc', Acc, epo)
        #writer.add_scalar('Acc_class', Acc_class, epo)
        writer.add_scalar('mIoU', mIoU, epo)        
        
        # 每次验证，根据新得出的miou指标来保存模型
        #global best_pred
        new_pred = mIoU
        if new_pred > best_pred:
            best_pred = new_pred
            torch.save(deeplabv3_model.state_dict(), 'models_v0304/deeplabv3_{}.pth'.format(epo))
        
        '''
        # 每5轮保存一下模型
        if np.mod(epo, 5) == 0:
            torch.save(deeplabv3_model, 'checkpoints_v0225/deeplabv3_model_{}.pt'.format(epo))
            print('saving checkpoints_v0225/deeplabv3_model_{}.pt'.format(epo))
        '''

if __name__ == "__main__":
    train(epoch=210)
