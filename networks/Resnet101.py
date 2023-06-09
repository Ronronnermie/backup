import torch.nn as nn
import torch


class Bottleneck(nn.Module):  #50、101、152层的ResNet
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  #在conv234 最后一层的卷积核个数是第一层的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,   #只是定义，不需要relu forward调用时再一一调用relu
                 groups=1, width_per_group=64): #？？？
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm3d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv3d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)   #stride 实线和虚线步距不一样 传入stride 进行调整
        self.bn2 = nn.BatchNorm3d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv3d(in_channels=width, out_channels=out_channel*self.expansion,   #该层卷积核个数变化，调用上面定义的expansion
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm3d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 为None 对应实线残差结构  不为None 对应虚线残差结构
            identity = self.downsample(x)  # 输入x通过下采样downsample 得到捷径分支的输出 identity

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,  # 残差结构
                 blocks_num,  # 使用残差结构的数目  是列表 如：34层 block_num=[3,4,6,3]
                 num_classes=4,  # 训练集分类个数
                 include_top=True,  # 方便在ResNet上搭建更复杂的网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top  # 将include_top传入类变量中
        self.in_channel = 64  # 输入特征矩阵的深度

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv3d(4, self.in_channel, kernel_size=7, stride=1,  # 卷积层1所用的卷积核个数是64 in_channel
                               padding=3, bias=False)  # padding=(f-1)/2  f=7
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # 为让输入的HW缩减为原来一半，

        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # conv_2对应的一系列残差结构/ 通过self._make_layer函数生成
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # conv_3
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # conv_4
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # conv_5
        # if self.include_top:
        #     self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # output size = (1, 1)
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # if self.include_top:
        #     x = self.avgpool(x)
        #     x = torch.flatten(x, 1)
        #     x = self.fc(x)

        return x

def resnet101(num_classes=4, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)