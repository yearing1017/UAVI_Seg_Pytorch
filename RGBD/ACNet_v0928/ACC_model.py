import torch
from torch import nn
from torch.nn import functional as F
import math
#import torch.utils.model_zoo as model_zoo
#from utils import utils
from torch.utils.checkpoint import checkpoint
from CC_Attention import CC_module as CrissCrossAttention

# 此代码基于原ACNet网络修改
# 将原有的通道注意力换为CC_module

BatchNorm2d = nn.BatchNorm2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ACNet(nn.Module):
    def __init__(self, num_class=4, pretrained=False):
        super(ACNet, self).__init__()

        layers = [3, 4, 6, 3]  #v0923版本 resnet50
        #layers = [3, 4, 23,3]  #v0925版本 resnet101
        #layers = [3, 8, 36, 3] #v0927版本 resnet152 
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
        #self.atten_rgb_0 = self.channel_attention(64)
        # 0927:不知如何设置out_channels
        self.atten_rgb_0 = RCCAModule(64, 64, num_class)
        self.atten_depth_0 = RCCAModule(64, 64, num_class)
        self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.atten_rgb_1 = RCCAModule(64*4, 64*4, num_class)
        self.atten_depth_1 = RCCAModule(64*4, 64*4, num_class)
        # self.conv_2 = nn.Conv2d(64*4, 64*4, kernel_size=1) #todo 用cat和conv降回通道数
        self.atten_rgb_2 = RCCAModule(128*4, 128*4, num_class)
        self.atten_depth_2 = RCCAModule(128*4, 128*4, num_class)
        self.atten_rgb_3 = RCCAModule(256*4, 256*4, num_class)
        self.atten_depth_3 = RCCAModule(256*4, 256*4, num_class)
        self.atten_rgb_4 = RCCAModule(512*4, 512*4, num_class)
        self.atten_depth_4 = RCCAModule(512*4, 512*4, num_class)

        self.inplanes = 64
        self.layer1_m = self._make_layer(block, 64, layers[0])
        self.layer2_m = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_m = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_m = self._make_layer(block, 512, layers[3], stride=2)

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

        self.out5_conv = nn.Conv2d(256, num_class, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128, num_class, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)

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
        #print('!!!!! ', rgb.shape)
        feature_rgb = self.atten_rgb_0(rgb)
        #print('!!!!!!!', feature_rgb.shape) # 64
        feature_depth = self.atten_depth_0(depth)
        # mul是乘法
        m0 = feature_rgb + feature_depth
        #print('m0:!!!!!!', m0.shape) # 64 
        rgb = self.maxpool(rgb)
        depth = self.maxpool_d(depth)
        m = self.maxpool_m(m0)
        #print('m:!!!!!!', m.shape) # 64
        # block 1
        rgb = self.layer1(rgb)
        depth = self.layer1_d(depth)
        m = self.layer1_m(m)
        #print('m:!!!!!!', m.shape) #256
        feature_rgb = self.atten_rgb_1(rgb)
        #print('!!!!!!!', feature_rgb.shape) #256
        feature_depth = self.atten_depth_1(depth)
        m1 = m + feature_rgb + feature_depth

        # block 2
        rgb = self.layer2(rgb)
        depth = self.layer2_d(depth)
        m = self.layer2_m(m1)
        #print('m:!!!!!!', m.shape) # 512
        feature_rgb = self.atten_rgb_2(rgb)
        #print('!!!!!!!', feature_rgb.shape) #512
        feature_depth = self.atten_depth_2(depth)
        m2 = m + feature_rgb + feature_depth

        # block 3
        rgb = self.layer3(rgb)
        depth = self.layer3_d(depth)
        m = self.layer3_m(m2) 
        #print('m:!!!!!!', m.shape) #1024
        feature_rgb = self.atten_rgb_3(rgb)
        #print('!!!!!!!', feature_rgb.shape) #1024
        feature_depth = self.atten_depth_3(depth)
        m3 = m + feature_rgb + feature_depth

        # block 4
        rgb = self.layer4(rgb)
        depth = self.layer4_d(depth)
        m = self.layer4_m(m3)
        #print('m:!!!!!!', m.shape) #2048
        feature_rgb = self.atten_rgb_4(rgb)
        #print('!!!!!!!', feature_rgb.shape) #2048
        feature_depth = self.atten_depth_4(depth)
        m4 = m + feature_rgb + feature_depth

        return m0, m1, m2, m3, m4  # channel of m is 2048

    def decoder(self, fuse0, fuse1, fuse2, fuse3, fuse4):
        agant4 = self.agant4(fuse4)
        # upsample 1
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)
        out = self.final_deconv(x)

        #if self.training:
            #return out, out2, out3, out4, out5
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

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1))
            

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_data = torch.rand((4,3,320,320))
    image_data = image_data.to(device)

    #label_data = torch.rand((4,1,320,320))
    #label_data = label_data.to(device)
    
    dsm_data = torch.rand((4,1,320,320))
    dsm_data = dsm_data.to(device)

    model = ACNet()
    model = model.to(device)
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
