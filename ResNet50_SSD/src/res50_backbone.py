# _*_ coding: utf-8 _*_

"""
  @Time    : 2024/1/5 10:53 
  @File    : res50_backbone.py
  @Project : ObjectDet
  @Author  : Saitama
  @IDE     : PyCharm
  @Des     : ResNet50 框架
"""
from torch import nn


class Bottleneck(nn.Module):
    """
    ResNet-50
    """

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        初始化
        @param in_channel:
        @param out_channel:
        @param stride:
        @param downsample: 下采样函数（浅层 18、34不需要，深层50、101、152需要）
        """
        super(Bottleneck, self).__init__()

        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        """
        正向传播
        @param x:
        @return:
        """
        downsample_result = None
        if self.downsample is not None:
            downsample_result = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if downsample_result is not None:
            out += downsample_result

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet网络
    """

    def __init__(self, block, block_nums, num_classes=1000, include_top=True):
        """

        @param block: 模块（BasicBlock[18,34] & BottleNeck[50,101,152]）
        @param block_nums: 每个block的重复次数
        @param num_classes: 预测类别数
        @param include_top:
        """
        super(ResNet, self).__init__()

        self.include_top = include_top

        # 进入BottleNeck的维度
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.in_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(block=Bottleneck, input_channel=64, repeat_times=block_nums[0], stride=1)
        self.layer2 = self._make_layer(block=Bottleneck, input_channel=128, repeat_times=block_nums[1], stride=2)
        self.layer3 = self._make_layer(block=Bottleneck, input_channel=256, repeat_times=block_nums[2], stride=2)
        self.layer4 = self._make_layer(block=Bottleneck, input_channel=512, repeat_times=block_nums[3], stride=2)

    def _make_layer(self, block, channel, repeat_times, stride):
        """
        基于block构建残差结构
        @param block:
        @param channel: 输入维度
        @param repeat_times: block的重复次数
        @param stride:
        @return:
        """
        # 因为使用的ResNet-50，所以一定有下采样
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel,
                          out_channels=channel * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = list()

        layers.append(block(in_channel=self.in_channel,
                            out_channel=channel,
                            downsample=downsample))

        self.in_channel = channel * block.expansion

        for _ in range(1, repeat_times):
            layers.append(block(in_channel=self.in_channel,
                                out_channel=channel))

        return nn.Sequential(*layers)
