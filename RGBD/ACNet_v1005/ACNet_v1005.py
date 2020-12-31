import torch
from torch import nn
from torch.nn import functional as F
import math
#import torch.utils.model_zoo as model_zoo
#from utils import utils
from torch.utils.checkpoint import checkpoint

# 论文中的图示网络代码；最终版本
# 此代码基于原ACNet网络修改
# 删除了中间的合并分支


class ACNet(nn.Module):
    def __init__(self, num_class=4, pretrained=False):
        super(ACNet, self).__init__()

        layers = [3, 4, 6, 3]  #v0923版本 resnet50
        #layers = [3, 4, 23,3]  #v0925版本 resnet101 
        block = Bottleneck
        transblock = TransBasicBlock
        # RGB image branch
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # use PSPNet extractors
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # depth image branch
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        # merge branch
        self.atten_rgb_0 = self.channel_attention(64)
        self.atten_depth_0 = self.channel_attention(64)
        #self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.atten_rgb_1 = self.channel_attention(64*4)
        self.atten_depth_1 = self.channel_attention(64*4)
        # self.conv_2 = nn.Conv2d(64*4, 64*4, kernel_size=1) #todo 用cat和conv降回通道数
        self.atten_rgb_2 = self.channel_attention(128*4)
        self.atten_depth_2 = self.channel_attention(128*4)
        self.atten_rgb_3 = self.channel_attention(256*4)
        self.atten_depth_3 = self.channel_attention(256*4)
        self.atten_rgb_4 = self.channel_attention(512*4)
        self.atten_depth_4 = self.channel_attention(512*4)

        self.inplanes = 64
        #self.layer1_m = self._make_layer(block, 64, layers[0])
        #self.layer2_m = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3_m = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4_m = self._make_layer(block, 512, layers[3], stride=2)

        # agant module
        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64*4, 64)
        self.agant2 = self._make_agant_layer(128*4, 128)
        self.agant3 = self._make_agant_layer(256*4, 256)
        self.agant4 = self._make_agant_layer(512*4, 512)

        #transpose layer
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        # final blcok
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv = nn.ConvTranspose2d(self.inplanes, num_class, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        #self.out5_conv = nn.Conv2d(256, num_class, kernel_size=1, stride=1, bias=True)
        #self.out4_conv = nn.Conv2d(128, num_class, kernel_size=1, stride=1, bias=True)
        #self.out3_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)
        #self.out2_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)

        self.conv_down_0 = conv_down(64*2, 64)
        self.conv_down_1 = conv_down(256*2, 256)
        self.conv_down_2 = conv_down(512*2, 512)
        self.conv_down_3 = conv_down(1024*2, 1024)
        self.conv_down_4 = conv_down(2048*2, 2048)
        self.conv_down_5 = conv_down(256*2, 256)
        self.conv_down_6 = conv_down(128*2, 128)
        self.conv_down_7 = conv_down(64*2, 64)
        self.conv_down_8 = conv_down(64*2, 64)

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    def encoder(self, rgb, depth):
        rgb = self.conv1(rgb)
        rgb = self.bn1(rgb)
        rgb = self.relu(rgb)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu_d(depth)
        # print('!!!!! ', rgb.shape)
        atten_rgb = self.atten_rgb_0(rgb)
        atten_depth = self.atten_depth_0(depth)

        m0_rgb = rgb.mul(atten_rgb)
        #print('m0_rgb:', m0_rgb.shape) # m0_rgb: torch.Size([4, 64, 160, 160])
        
        m0_depth = depth.mul(atten_depth)
        #print('m0_depth:', m0_depth.shape) # m0_depth: torch.Size([4, 64, 160, 160])
        # mul是乘法
        #m0 = m0_rgb + m0_depth
        m0 = torch.cat((m0_rgb,m0_depth),1)
        m0 = self.conv_down_0(m0) 
        #print('m0:!!!!!!', m0.shape) # m0:!!!!!! torch.Size([4, 64, 160, 160])

        m0_rgb = self.maxpool(m0_rgb)
        #print('m0_rgb:', m0_rgb.shape) # m0_rgb: torch.Size([4, 64, 80, 80])
        m0_depth = self.maxpool_d(m0_depth)
        #print('m0_depth:', m0_depth.shape) # m0_depth: torch.Size([4, 64, 80, 80])
        #m = self.maxpool_m(m0)

       

        # block 1
        m1_rgb = self.layer1(m0_rgb)
        m1_depth = self.layer1_d(m0_depth)
        #m = self.layer1_m(m)

        atten_rgb = self.atten_rgb_1(m1_rgb)
        atten_depth = self.atten_depth_1(m1_depth)
        m1_rgb = m1_rgb.mul(atten_rgb)
        m1_depth = m1_depth.mul(atten_depth)

        #m1 = m1_rgb + m1_depth
        m1 = torch.cat((m1_rgb,m1_depth),1)
        m1 = self.conv_down_1(m1) 
        
        # block 2
        m2_rgb = self.layer2(m1_rgb)
        m2_depth = self.layer2_d(m1_depth)
        #m = self.layer2_m(m1)

        atten_rgb = self.atten_rgb_2(m2_rgb)
        atten_depth = self.atten_depth_2(m2_depth)
        m2_rgb = m2_rgb.mul(atten_rgb)
        m2_depth = m2_depth.mul(atten_depth)
        #m2 = m2_rgb + m2_depth
        m2 = torch.cat((m2_rgb,m2_depth),1)
        m2 = self.conv_down_2(m2) 

        # block 3
        m3_rgb = self.layer3(m2_rgb)
        m3_depth = self.layer3_d(m2_depth)
        #m = self.layer3_m(m2)

        atten_rgb = self.atten_rgb_3(m3_rgb)
        atten_depth = self.atten_depth_3(m3_depth)
        m3_rgb = m3_rgb.mul(atten_rgb)
        m3_depth = m3_depth.mul(atten_depth)

        #m3 = m3_rgb + m3_depth
        m3 = torch.cat((m3_rgb,m3_depth),1)
        m3 = self.conv_down_3(m3) 
        
        
        # block 4
        m4_rgb = self.layer4(m3_rgb)
        m4_depth = self.layer4_d(m3_depth)
        #m = self.layer4_m(m3)

        atten_rgb = self.atten_rgb_4(m4_rgb)
        atten_depth = self.atten_depth_4(m4_depth)
        m4_rgb = m4_rgb.mul(atten_rgb)
        m4_depth = m4_depth.mul(atten_depth)
        #m4 = m4_rgb + m4_depth
        m4 = torch.cat((m4_rgb,m4_depth),1)
        m4 = self.conv_down_4(m4) 

        return m0, m1, m2, m3, m4  # channel of m is 2048

    def decoder(self, fuse0, fuse1, fuse2, fuse3, fuse4):
        agant4 = self.agant4(fuse4) 
        #print('agant4:',agant4.shape) # agant4: torch.Size([4, 512, 10, 10])
        # upsample 1
        x = self.deconv1(agant4)
        #print('x1',x.shape)  # x1 torch.Size([4, 256, 20, 20])
        #print('fuse3',fuse3.shape) # fuse3 torch.Size([4, 1024, 20, 20])

        #x = x + self.agant3(fuse3)
        x = torch.cat((x,self.agant3(fuse3)),1)
        x = self.conv_down_5(x)
        #print('self.agant3(fuse3):',self.agant3(fuse3).shape) # self.agant3(fuse3): torch.Size([4, 256, 20, 20])
        #print('x1:',x.shape)  # x1: torch.Size([4, 256, 20, 20])
        
        
        # upsample 2
        x = self.deconv2(x)
        #print('x2:',x.shape) # x2: torch.Size([4, 128, 40, 40])
        #x = x + self.agant2(fuse2)
        x = torch.cat((x,self.agant2(fuse2)),1)
        x = self.conv_down_6(x)
        #print('fuse2',fuse2.shape) # fuse2 torch.Size([4, 512, 40, 40])
        #print('self.agant2(fuse2):',self.agant2(fuse2).shape) # self.agant2(fuse2): torch.Size([4, 128, 40, 40])
        #print('x2:',x.shape) # x2: torch.Size([4, 128, 40, 40])

        # upsample 3
        x = self.deconv3(x)
        #print('x3:',x.shape) # x3: torch.Size([4, 64, 80, 80])
        #x = x + self.agant1(fuse1)
        x = torch.cat((x,self.agant1(fuse1)),1)
        x = self.conv_down_7(x)
        #print('fuse1',fuse1.shape) # fuse1 torch.Size([4, 256, 80, 80])
        #print('self.agant1(fuse1):',self.agant1(fuse1).shape) # self.agant1(fuse1): torch.Size([4, 64, 80, 80])
        #print('x3:',x.shape) # x3: torch.Size([4, 64, 80, 80])


        # upsample 4
        x = self.deconv4(x)
        #print('x4:',x.shape) # x4: torch.Size([4, 64, 160, 160])
        #print('fuse0:',fuse0.shape) # fuse0: torch.Size([4, 64, 160, 160])
        #print('self.agant0(fuse0):',self.agant0(fuse0).shape) # self.agant0(fuse0): torch.Size([4, 64, 160, 160])
        #x = x + self.agant0(fuse0)
        x = torch.cat((x,self.agant0(fuse0)),1)
        x = self.conv_down_8(x)
        
        
        #print('x4:',x.shape)
        # final
        x = self.final_conv(x)
        #print('x:',x.shape) # x: torch.Size([4, 64, 160, 160])
        out = self.final_deconv(x)

        # 不使用论文的求loss方法，使用原始尺寸上采样的结果作为输出
        return out

    def forward(self, rgb, depth, phase_checkpoint=False):
        fuses = self.encoder(rgb, depth)
        m = self.decoder(*fuses)
        return m

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    # SE通道注意力
    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        # bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid() # todo modify the activation function

        return nn.Sequential(*[pool, conv, activation])

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_down(in_planes, out_planes, stride=1):
    '1x1卷积来减少channle'
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                        bias=False)

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

if __name__ == '__main__':
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    image_data = torch.rand((4,3,320,320))
    #image_data = image_data.to(device)

    #label_data = torch.rand((4,1,320,320))
    #label_data = label_data.to(device)
    
    dsm_data = torch.rand((4,1,320,320))
    #dsm_data = dsm_data.to(device)

    model = ACNet()
    #model = model.to(device)
    '''
    out_arr = model(image_data, dsm_data)
    for out in out_arr:
        print(out.shape)
    '''
    out = model(image_data,dsm_data)
    print(out.shape)
'''
原始代码打印的输出：
torch.Size([4, 4, 320, 320])
torch.Size([4, 4, 160, 160])
torch.Size([4, 4, 80, 80])
torch.Size([4, 4, 40, 40])
torch.Size([4, 4, 20, 20])

'''
