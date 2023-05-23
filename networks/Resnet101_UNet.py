from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Resnet101 import resnet101
from torchsummary import summary


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )


class Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, int(in_channels // 2))
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # [N, C, H, W]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Resnet101_Unet(nn.Module):

    def __init__(self, in_channels: int = 4, num_classes: int = 4, base_c: int = 64, bilinear: bool = False):
        super(Resnet101_Unet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.resnet = resnet101()

        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        # HWD
        self.maxpool = self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # encode
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        # resnet最后一层avgpool用卷积层代替
        self._avgpool = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),   #(2048,5,5)
            nn.Conv3d(kernel_size=1, in_channels=2048, out_channels=4096, padding=0),
            nn.BatchNorm3d(4096),
            nn.ReLU(),
            nn.Conv3d(kernel_size=1, in_channels=4096, out_channels=4096, padding=0),
            nn.BatchNorm3d(4096),
            nn.ReLU())

        # decode
        self.decoder4 = Up(2048+2048, 2048, 2048)
        self.decoder3 = Up(1024+1024, 1024, 1024)
        self.decoder2 = Up(512+512, 512, 512)
        self.decoder1 = Up(256+256, 256, 256)
        self.decoder0 = Up(128+64, 64, 64)
        self.out_conv = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.maxpool(x1)
        # encode
        encode1 = self.encoder1(x2)
        encode2 = self.encoder2(encode1)
        encode3 = self.encoder3(encode2)
        encode4 = self.encoder4(encode3)

        # Bottleneck
        _Bottleneck = self._avgpool(encode4)

        # decode
        decode4 = self.decoder4(_Bottleneck, encode4)
        decode3 = self.decoder3(decode4, encode3)
        decode2 = self.decoder2(decode3, encode2)
        decode1 = self.decoder1(decode2, encode1)
        decode0 = self.decoder0(decode1, x1)

        out_conv = self.out_conv(decode0)

        return out_conv

# 测试网络
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Resnet101_Unet(in_channels=4, num_classes=4).to(device)
# summary(model, input_size=(3, 112, 112))


if __name__ == '__main__':
    x = torch.randn(1, 4, 160, 160, 128)
    net = Resnet101_Unet(in_channels=4, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)
