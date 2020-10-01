import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from AC_data import train_dataloader,val_dataloader
#from ACNet_model import ACNet
from ACC_model import ACNet
from tensorboardX import SummaryWriter
from MIouv0217 import Evaluator

def train(epoch=45):
    # 创建指标计算对象
    evaluator = Evaluator(4)

    # 定义最好指标miou数值，初始化为0
    best_pred = 0.0
    writer = SummaryWriter('tblog/acnet_v0928')
    # 指定第二块gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 模型建立
    ac_model = ACNet()
    ac_model = ac_model.to(device)
    
    # 加入权重，减轻样本不平衡情况
    weight = torch.from_numpy(np.array([0.25,0.88,0.90,0.97])).float()
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=weight).to(device) # CrossEntropyLoss适用多分类
    optimizer = optim.Adam(ac_model.parameters(), lr=1e-3)

    for epo in range(epoch):
        # 每个epoch都要记录5次交叉验证的train_loss和val_loss，最后除5
        train_loss = 0
        val_loss = 0
        val_acc = 0
        val_miou = 0
        for i in range(5):
            # 训练部分
            ac_model.train()
            for index, (image, dsm_data, label) in enumerate(train_dataloader[i]):
                image = image.to(device)
                dsm_data = dsm_data.to(device) 
                label = label.to(device)
                # 此句提前
                optimizer.zero_grad()
                output = ac_model(image, dsm_data)
                loss = criterion(output, label)
                
                loss.backward()
                iter_loss = loss.item() # 取出数值
                train_loss += iter_loss
                optimizer.step()

                if np.mod(index, 18) == 0:
                    line = "epoch {}_{}, {}/{},train loss is {}".format(epo, i, index, len(train_dataloader[i]), iter_loss)
                    print(line)
                    # 写到日志文件
                    with open('log/logs_acnet_0928.txt', 'a') as f :
                        f.write(line)
                        f.write('\r\n')

            # 验证部分
            ac_model.eval()
            with torch.no_grad():
                for index, (image, dsm_data, label) in enumerate(val_dataloader[i]):
                    image = image.to(device)
                    dsm_data = dsm_data.to(device)
                    label = label.to(device)
                    
                    output = ac_model(image, dsm_data)
                    loss = criterion(output, label)
                    optimizer.zero_grad()
                    iter_loss = loss.item()
                    val_loss += iter_loss
                    # 记录相关指标数据
                    pred = output.cpu().numpy()
                    label = label.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    evaluator.add_batch(label, pred)
                Acc = evaluator.Pixel_Accuracy()
                mIoU = evaluator.Mean_Intersection_over_Union()
                val_acc += Acc
                val_miou += mIoU
                evaluator.reset() # 该5次求指标，每次求之前先清零
        line_epoch = "epoch train loss = %.3f, epoch val loss = %.3f" % (train_loss/len(train_dataloader[i])/5, val_loss/len(val_dataloader[i])/5)
        print(line_epoch)
        with open('log/logs_acnet_0928.txt', 'a') as f :
            f.write(line_epoch)
            f.write('\r\n')
        
        
        # tensorboard记录
        writer.add_scalar('train_loss', train_loss/len(train_dataloader[i])/5, epo)
        writer.add_scalar('val_loss', val_loss/len(val_dataloader[i])/5, epo)
        writer.add_scalar('val_Acc', val_acc/5, epo)
        #writer.add_scalar('Acc_class', Acc_class, epo)
        writer.add_scalar('val_mIoU', val_miou/5, epo)        
        
        # 每次验证，根据新得出的miou指标来保存模型
        #global best_pred
        new_pred = val_miou/5
        if new_pred > best_pred:
            best_pred = new_pred
            torch.save(ac_model.state_dict(), 'models_acnet_v0928/acnet_0928_{}.pth'.format(epo))

        print("Training completed ")
if __name__ == "__main__":
    train()    