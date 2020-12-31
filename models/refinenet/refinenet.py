import torch
import torch.nn as nn
import math
import torchvision.models as models
BatchNorm2d = nn.BatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, bn_momentum=0.0003):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
    
    def _sum_each(self, x, y):
        assert(len(x)==len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

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

        out = out + residual      
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
        self.inplanes = 128
        self.is_fpn = is_fpn
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilation[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1 if dilation[1]!=1 else 2, dilation=dilation[1], bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1 if dilation[2]!=1 else 2, dilation=dilation[2], bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if dilation[3]!=1 else 2, dilation=dilation[3], bn_momentum=bn_momentum)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, bn_momentum=0.0003):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = True, momentum=bn_momentum))

        layers = []
        #generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=multi_grid, bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=multi_grid, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x, start_module=1, end_module=5):
        if start_module <= 1:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            start_module = 2
        features = []
        for i in range(start_module, end_module+1):
            x = eval('self.layer%d'%(i-1))(x)
            features.append(x)

        if self.is_fpn:
            if len(features) == 1:
                return features[0]
            else:
                return tuple(features)
        else:
            return x


def get_resnet50(num_classes=4, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
    model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes, dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn)
    return model


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        # two pool-block
        for i in range(1, 3):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),   # obtain the raw feature map size
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 3):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x

class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)


class BaseRefineNet4Cascade(nn.Module):
    def __init__(self, input_channel, input_size,  refinenet_block, num_classes=1, features=256, resnet_factory=models.resnet50, bn_momentum = 0.01, pretrained=True, freeze_resnet=False):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__()

        input_channel = input_channel
        input_size = input_size

        self.Resnet101 = get_resnet50(num_classes=0, bn_momentum=bn_momentum)

        self.layer1 = nn.Sequential(self.Resnet101.conv1, self.Resnet101.bn1, self.Resnet101.relu1,
                                    self.Resnet101.conv2, self.Resnet101.bn2, self.Resnet101.relu2,
                                    self.Resnet101.conv3, self.Resnet101.bn3, self.Resnet101.relu3,
                                    self.Resnet101.maxpool, self.Resnet101.layer1)

        self.layer2 = self.Resnet101.layer2
        self.layer3 = self.Resnet101.layer3
        self.layer4 = self.Resnet101.layer4

        # freeze the resnet parameters, default is false
        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # modify the feature maps from each stage of RenNet, modify their channels
        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)     # here, 2*fetures means we use two same stage-4 features as input

        self.refinenet4 = RefineNetBlock(2 * features,
                                         (2 * features, math.ceil(input_size // 32)))
        self.refinenet3 = RefineNetBlock(features,
                                         (2 * features, input_size // 32),
                                         (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(features,
                                         (features, input_size // 16),
                                         (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(features, (features, input_size // 8),
                                         (features, input_size // 4))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

    def forward(self, x):
        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        # modify the number of channel
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        out = nn.functional.interpolate(out, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return out

    # def named_parameters(self):
    #     """Returns parameters that requires a gradident to update."""
    #     return (p for p in super().named_parameters() if p[1].requires_grad)


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):
    def __init__(self,
                 input_shape,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet50,
                 bn_momentum = 0.01,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            RefineNetBlockImprovedPooling,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            bn_momentum = bn_momentum,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


class RefineNet4Cascade(BaseRefineNet4Cascade):
    def __init__(self,
                 input_size,
                 input_channel=3,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet50,
                 bn_momentum = 0.01,
                 pretrained=True,
                 freeze_resnet=False):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_channel (int): channe num
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_channel,
            input_size,
            RefineNetBlock,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            bn_momentum = bn_momentum,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


def get_refinenet(input_size, num_classes, features=256, bn_momentum=0.01, pretrained=False):
    resnet50 = get_resnet50(num_classes=num_classes, bn_momentum=bn_momentum)
    return RefineNet4Cascade(input_size, num_classes=num_classes, resnet_factory=resnet50,
                             features=features, bn_momentum=bn_momentum, pretrained=pretrained)

    

if __name__ == '__main__':
    num_classes = 4
    in_batch, inchannel, in_h, in_w = 4, 3, 320, 320
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = get_refinenet(input_size=in_h, num_classes=num_classes, pretrained=False)
    out = net(x)
    print(out.shape)
