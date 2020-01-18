import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 基于F.conv2d自己建的Conv2d类，其中F.conv2d仅仅只是卷积操作，而nn.Conv2d是卷积层类
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# ResNet中的block类型，指的是1x1,3x3,1x1三种卷积混合的模式，采用先降维再升维，降低计算复杂度
class Bottleneck(nn.Module):
    expansion = 4 # 在block最后升维的倍数，恢复原来的通道数
    # 这里的planes不再是网络中的输出通道数，而是在block中降维的输出通道数
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
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
        # 此处的downsample利用1x1卷积来改变通道数，使残差块的连接可以直接相加
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# deeplabv3的ASPP模块
class ASPP(nn.Module):
    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C # 进入aspp的通道数
        self._depth = depth # filter的个数
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        # 第一个1x1卷积
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        # aspp中的空洞卷积，rate=6，12，18
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        # 对最后一个特征图进行全局平均池化，再feed给256个1x1的卷积核，都带BN
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        # 先上采样双线性插值得到想要的维度，再进入下面的conv
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        # 打分分类
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        # 上采样：双线性插值使x得到想要的维度
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        # 经过aspp之后，concat之后通道数变为了5倍
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


# 基于ResNet的deeplabv3
class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes, num_groups=None, weight_std=False, beta=False):
        self.inplanes = 64 # 控制残差块的输入通道数 planes:输出通道数
        # nn.BatchNorm2d和nn.GroupNorm两种不同的归一化方法
        self.norm = lambda planes, momentum=0.5: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        self.conv = Conv2d if weight_std else nn.Conv2d
        super(ResNet, self).__init__()

        if not beta:
            # 整个ResNet的第一个conv
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            # 第一个残差模块的conv
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 建立残差块部分
        self.layer1 = self._make_layer(block, 64,  block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        # block4开始为dilation空洞卷积
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, dilation=2)
        # aspp,512 * block.expansion是经过残差模块的输出通道数
        self.aspp = ASPP(512 * block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)
        # 遍历模型进行初始化
        for m in self.modules():
            if isinstance(m, self.conv):        #isinstance：m类型判断    若当前组件为 conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  #正太分布初始化
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm): #若为batchnorm
                m.weight.data.fill_(1)          #weight为1
                m.bias.data.zero_()             #bias为0

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        # stride!=1 代表后续残差块中有stride=2，尺寸大小改变，所以第一个残差块中的stride也该用来修改尺寸
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )
        # laysers 存放产生的残差块，最后根据此列表进行生成网络
        layers = []
        # 在多个残差块中，只有第一个残差块的输入输出通道不一致，所以先单独添加带downsample的block
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)


    def forward(self, x):
        # x.shape:[batch_size, channels, H, w]
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x


# 实例化模型
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # [3,4,6,3]对应block_num,残差块的数量
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=4,num_groups=num_groups, weight_std=weight_std, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        if num_groups and weight_std:
            pretrained_dict = torch.load('data/R-101-GN-WS.pth.tar')
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            assert len(overlap_dict) == 312
        elif not num_groups and not weight_std:
            pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__ == "__main__":
    net = resnet50()
    x = torch.rand((4,3,640,640))
    output = net(x)
    print(output.shape) #(4,4,640,640)
'''
# 如下测试，不会执行最后的unsample步骤，所以shape不和原来一样
x = torch.rand((4,3,640,640))
for name, layer in net.named_children():
    x = layer(x)
    print(name, ' output shape:\t', x.shape)    
'''
'''
上述测试的输出
conv1  output shape:	 torch.Size([4, 64, 320, 320])
bn1  output shape:	 torch.Size([4, 64, 320, 320])
relu  output shape:	 torch.Size([4, 64, 320, 320])
maxpool  output shape:	 torch.Size([4, 64, 160, 160])
layer1  output shape:	 torch.Size([4, 256, 160, 160])
layer2  output shape:	 torch.Size([4, 512, 80, 80])
layer3  output shape:	 torch.Size([4, 1024, 40, 40])
layer4  output shape:	 torch.Size([4, 2048, 40, 40])
aspp  output shape:	 torch.Size([4, 4, 40, 40])
'''
