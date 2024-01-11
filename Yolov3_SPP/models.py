# _*_ coding: utf-8 _*_

"""
  @Time    : 2024/1/10 15:15 
  @File    : model.py
  @Project : ObjectDet
  @Author  : Saitama
  @IDE     : PyCharm
  @Des     : Yolov3_SPP 模型
"""
from typing import List

from torch import nn
from utils.parse_config import parse_model_cfg


def create_modules(module_defs: List, img_size):
    """
    根据传入的配置信息，一层层搭建网络结构
    :param module_defs:
    :param img_size:
    :return:
    """
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除第一层配置（net）
    module_defs.pop(0)
    output_filters = [3]  # input channel
    module_list = nn.ModuleList()  # 保存网络结构
    routs = list()  # 统计哪些特征层的输出会被后续的层使用
    yolo_index = -1

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        # 卷积层
        if module_def["type"] == "convolutional":
            output_channel = module_def["filters"]  # 卷积核的个数 = 输出维度
            kernel_size = module_def["size"]
            stride = module_def["stride"]
            padding = kernel_size // 2 if module_def["pad"] else 0
            bn = module_def["batch_normalize"]  # 1 for use, 0 for not
            activation_func = module_def["activation"]

            if isinstance(kernel_size, int):
                modules.add_module("Conv2d", nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=not bn))
            else:
                raise TypeError("Conv2d kernel size should be int type")

            if bn:
                # 有BN层
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(output_channel))
            else:
                # 没有BN层，即为yolo的三个predictor !!!
                routs.append(i)

            if activation_func == "leaky":
                # 激活函数[除了yolo的三个predictor，其他的卷积层使用的激活函数都是leaky]
                modules.add_module("activation", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            else:
                pass

        # 池化层【只在SPP中用到】
        elif module_def["type"] == "maxpool":
            kernel_size = module_def["size"]
            stride = module_def["stride"]
            padding = (kernel_size - 1) // 2
            modules = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

        # 上采样层
        elif module_def["type"] == "upsample":
            # 采样率（倍数）
            ratio = module_def["stride"]
            modules = nn.Upsample(scale_factor=ratio)

        module_list.add_module(modules)


def get_yolo_layers(self):
    pass


class Darknet(nn.Module):
    """
    Yolov3_SPP 中采用的Darknet
    """

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        """
        初始化函数
        :param cfg: 模型配置文件路径
        :param img_size: 只在导出ONNX模型时启用
        :param verbose: 是否打印模型的详细信息
        """
        super(Darknet, self).__init__()

        # 输入尺寸（只在导出ONNX模型时使用）
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析配置文件，得到模型的配置信息
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的配置信息，一层一层搭建网络
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YoloLayer层的索引
        self.yolo_layers = get_yolo_layers(self)

        # 打印模型信息
        self.info(verbose)

    def forward(self, x, verbose=False):
        """
        正向传播
        :param x:
        :param verbose:
        :return:
        """
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        pass
