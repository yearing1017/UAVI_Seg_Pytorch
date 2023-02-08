import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DoubleConv, self).__init__()
		# 构建一个“容器网络”
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self,x):
        # print(x.size())
		return self.double_conv(x)

class Down(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels,out_channels)
		)
	def forward(self,x):
		return self.maxpool_conv(x)

class Up(nn.Module):
	def __init__(self, in_channels,out_channels,bilinear=False):
		super().__init__()

		# 双线性插值上采样和转置卷积上采样两种方法
		if bilinear:
			# scale_factor=2代表输出是输入大小的两倍，bilinear代表双线性插值，类似还有最邻近插值等
			self.up = nn.Unsample(scale_factor=2,mode='bilinear',align_corners=False)
		else:
			# 下面的in_channles // 2 是代表了论文中“，每次使用反卷积都使特征通道减半，特征图大小加倍”
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
		# 上采样之后，因为有cat拼接，所以in_channles又变回了两倍，见图
		self.conv = DoubleConv(in_channels,out_channels)

	def forward(self,x1,x2):
		x1 = self.up(x1)
		# input [B,C,h,w]
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		# 扩充张量的边界
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# concat两边的特征图
		x = torch.cat([x2,x1],dim=1)
		return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)

# UNet
class UNet(nn.Module):
    def __init__(self, in_channels, in_classes, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.in_classes = in_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, in_classes)

    def forward(self, x):
        #print("unet_diyiceng")
        x1 = self.inc(x)
        #print("unet_1jieshu")
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
