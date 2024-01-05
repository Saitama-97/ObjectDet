# _*_ coding: utf-8 _*_

"""
  @Time    : 2024/1/5 13:33 
  @File    : ssd_model.py
  @Project : ObjectDet
  @Author  : Saitama
  @IDE     : PyCharm
  @Des     : SSD 模型
"""
import torch
from torch import nn

from ResNet50_SSD.src.res50_backbone import resnet50


class Backbone(nn.Module):
    """
    Backbone[ResNet50的前半段，用于特征提取]
    """

    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        # 实例化 ResNet50
        net = resnet50()
        # SSD的六个预测层的输出维度
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        # 如果有预训练模型则载入
        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        # 保留ResNet50中conv5_x之前的部分，作为特征提取器
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        # 将conv4_x中的第一个block的stride设置为1
        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        """
        正向传播过程
        @param x:
        @return:
        """
        x = self.feature_extractor(x)

        return x


class SSD300(nn.Module):
    """
    SSD = backbone + 五个额外添加层（用于不同尺寸）
    """

    def __init__(self, backbone=None, num_classes=21):
        """
        @param backbone: 此处backbone采用的是ResNet50，用于特征提取
        @param num_classes: 预测类别数
        """
        super(SSD300, self).__init__()

        if backbone is None:
            raise Exception("backbone is None")

        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone has no attribute: out_channels")

        # 预测类别数
        self.num_classes = num_classes

        # 特征提取器【ResNet50】
        self.feature_extractor = backbone

        # 额外添加层，得到一系列用于不同尺寸的特征提取器
        self.additional_blocks = None
        self.build_additional_features(backbone.out_channels)

        # 每个特征层生成的default_box个数
        self.num_default = [4, 6, 6, 6, 4, 4]

        # 位置预测器
        location_extractors = []
        # 置信度预测器
        confidence_extractors = []

        for nd, oc in zip(self.num_default, self.feature_extractor.out_channels):
            # 每个特征层生成对应数量的default_box，每个box由4个（两组）坐标确定
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            # 每个特征层生成对应数量的default_box，每个box具有属于各个类别的概率
            confidence_extractors.append((nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1)))

        # 位置预测器
        self.loc = nn.ModuleList(location_extractors)
        # 置信度预测器
        self.conf = nn.ModuleList(confidence_extractors)

    def build_additional_features(self, input_size):
        """
        额外添加层，构建用于不同尺寸的特征提取器
        @param input_size: 不同尺寸的输出维度[1024, 512, 512, 256, 256, 256]
        @return:
        """
        additional_blocks = []
        # 后五层的中间维度
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layers = nn.Sequential(
                nn.Conv2d(in_channels=input_ch,
                          out_channels=middle_ch,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=middle_ch,
                          out_channels=output_ch,
                          kernel_size=3,
                          stride=stride,
                          padding=padding,
                          bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True)
            )
            additional_blocks.append(layers)

        self.additional_blocks = nn.ModuleList(additional_blocks)

    def forward(self, image, targets=None):
        """
        SSD 正向传播过程
        @param image: 输入图片
        @param targets
        @return:
        """
        # 先由ResNet50提取出特征
        x = self.feature_extractor(image)

        detection_features = list()

        # 存储ResNet以及另外五个额外特征层的输出特征
        # [38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256]
        detection_features.append(x)

        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        # [38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4]
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        # 如果是训练模式
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            # gt 位置
            gt_bboxes = targets["bboxes"]
            gt_bboxes = gt_bboxes.transpose(1, 2).contiguous()
            # gt 类别
            gt_labels = targets["labels"]

            # 根据预测结果和gt计算loss
            loss = self.compute_loss(locs, confs, gt_bboxes, gt_labels)
            return {"total_losses": loss}

        # 如果是验证模式，则进行后处理，然后返回
        # 后处理：将预测结果叠加到default box，得到最终的bbox，并进行非最大值抑制，滤除重叠框
        results = self.postprocess(locs, confs)
        return results

    def bbox_view(self, features, loc_extractor, conf_extractor):
        """
        获取所有预测特征层上的位置参数以及置信度参数
        @param features:
        @param loc_extractor:
        @param conf_extractor:
        @return:
        """
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, h, w] -> [batch, 4, -1] ps:-1表示由系统补全
            locs.append(l(f).view(f.size(0), 4, -1))
            # [batch, n*classes, h, w] -> [batch, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        # contiguous将底层数据在存储上连续
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()

        return locs, confs

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameter():
                if param.dim() > 1:
                    nn.init.xavier_uniform(param)
        pass
