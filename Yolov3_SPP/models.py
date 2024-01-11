# _*_ coding: utf-8 _*_

"""
  @Time    : 2024/1/10 15:15 
  @File    : model.py
  @Project : ObjectDet
  @Author  : Saitama
  @IDE     : PyCharm
  @Des     : Yolov3_SPP 模型
"""
import math
from typing import List

import torch
from torch import nn
from utils.parse_config import parse_model_cfg


class FeatureConcat(nn.Module):
    """
    将多个特征矩阵在channel维度进行concatenate拼接
    """

    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # 输入的网络结构
        self.multiple = len(layers) > 1  # 判断输入的网络结构是否多层

    def forward(self, x, outputs):
        """
        正向传播过程
        :param x:
        :param outputs:
        :return:
        """
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    """
    将多个特征矩阵的值进行融合(add操作)
    """

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices，如 [-3]
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers 融合的特征矩阵个数[2]
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):  # range(1)
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[
                self.layers[i]]  # feature to add，如 a = outputs[-3]
            na = a.shape[1]  # feature channels

            # Adjust channels
            # 根据相加的两个特征矩阵的channel选择相加方式
            if nx == na:  # same shape 如果channel相同，直接相加
                x = x + a
            elif nx > na:  # slice input 如果channel不同，将channel多的特征矩阵砍掉部分channel保证相加的channel一致
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class YOLOLayer(nn.Module):
    """
    对Yolo的输出进行处理
    """

    def __init__(self, anchors, nc, img_size, stride):
        """
        初始化函数
        :param anchors: 待生成anchor的配置信息
        :param nc: 类别数
        :param img_size:
        :param stride: 特征图相对于原图的缩放比例
        """
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # 特征图相对于原图的缩放比例
        self.na = len(anchors)  # number of anchors
        self.nc = nc  # 类别数
        self.no = nc + 5  # number of outputs
        self.nx, self.ny, self.ng = 0, 0, (0, 0)

        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride

        # batch_size, na, grid_h, grid_w, wh
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息，并生成新的grids坐标
        :param ng: 特征矩阵的宽度、高度
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.Tensor(ng, dtype=torch.float)

        # 构建每个cell处的anchor的xy偏移量（在feature map上）
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na ,grid_w, grid_h,
            # 通俗的说，self.grid存储着 特征图坐标系 中，每个cell左上角的绝对坐标
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, x):
        """
        正向传播过程
        :param x: 预测的参数
        :return:
        """
        # batch_size, predict_param(255), grid_height, grid_width
        bs, _, ny, nx = x.shape
        if (self, nx, self.ny) != (nx, ny) or self.grid is None:
            self.create_grids((nx, ny), x.device)

        # view [batch_size, anchor_num, output_num, grid_y, grid_x]
        # view [batch_size, 255, 13, 13] -> [batch_size, 3, 85, 13, 13]
        # permute [ batch_size, anchor_num, grid_y, grid_x, output_num]
        # permute [ batch_size, 3, 13, 13, 85]
        # permute：改变维度信息
        # contiguous：使其内存连续
        x = x.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return x
        else:
            # [batch_size, anchor_num, grid_y, grid_x, output_num]
            # [1, 3, 13, 13, 85]
            # output_num = xywh + obj + classes = 4 + 1 + 80 = 85
            io = x.clone()
            # 基于预测得到的anchor的中心点在特征图坐标系中相对于cell的偏移量 A
            # 以及特征图坐标系中cell的左上角绝对坐标 B
            # 可以得到anchor在特征图坐标系中的绝对坐标 C
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            # 基于预测得到的anchor的width和height A
            # 以及同比例缩放的预先设定的anchor尺寸 B
            # 可以得到anchor在特征图坐标系中的width 和 height C
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            # 得到预测anchor在特征图坐标系中的中心点绝对坐标以及预测框的宽高，就可以换算成原图中的尺度
            io[..., 4] *= self.stride

            # [1, 3, 13, 13, 85] -> [1, 507, 85]
            return io.views(bs, -1, self.no), x


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
    output_filters = [3]  # 保存输出维度的每一次变动，初始为3【RGB图像】
    module_list = nn.ModuleList()  # 保存网络结构
    routs = list()  # routs：记录后续会被使用（特征融合、拼接）的模块的索引
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

        elif module_def["type"] == "route":
            # [-2]
            # [-1, 61]
            # [-1, -3, -5, -6]
            layers = module_def["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        # shortcut连接
        elif module_def["type"] == "shortcut":
            fro = module_def["from"]  # 与前面的哪一层进行融合[索引]，如[-3]
            activation = module_def["activation"]  # 激活函数（一般都是线性）
            filters = output_filters[-1]  # 上一层的输出维度
            routs.append(i + fro[0])  # 将需要融合的层加入
            modules = WeightedFeatureFusion(layers=fro)

        # yolo层
        elif module_def["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo layer [0, 1, 2]
            stride = [32, 16, 8]  # 特征图相对于原图的缩放比例

            modules = YOLOLayer(anchors=module_def["anchors"][module_def["mask"]],  # 使用的anchor list
                                nc=module_def["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            try:  # Focal loss, pass
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                j = -1
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)

        # 将每个模块添加到module_list
        module_list.append(modules)
        # 将channel发生的变化记录下来
        output_filters.append(filters)

    # 记录哪些模块会被后面用到
    routs_binary = [False] * len(module_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


def get_yolo_layers(self):
    """
    获取搭建的网络中三个"YOLO Layer"模块对应的索引
    :param self:
    :return:
    """
    return [i for i, module in enumerate(self.module_list) if module.__class__.__name__ == "YOLOLayer"]


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
