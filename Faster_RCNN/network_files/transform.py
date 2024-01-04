# _*_ coding: utf-8 _*_

"""
  @Time    : 2024.01.03 15:31
  @File    : transform.py
  @Project : ObjectDet
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 将读入的图像进行标准化处理
"""
import math
from typing import List, Dict, Tuple

import torch
import torchvision
from torch import nn, Tensor
from torch.nn.functional import interpolate

from Faster_RCNN.network_files.image_list import ImageList


class GeneralizedRCNNTransform(nn.Module):
    """
    对数据进行标准化，缩放，打包成batch等处理部分
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        """
        初始化
        :param min_size: 指定图像的最小边长范围
        :param max_size: 指定图像的最大边长范围
        :param image_mean: 指定图像在标准化过程中的均值
        :param image_std: 指定图像在标准化过程中的方差
        """
        super(GeneralizedRCNNTransform, self).__init__()

        # 如果min_size不是list或tuple类型，则转换成tuple类型
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        """
        对输入图像进行标准化处理
        :param image:
        :return:
        """
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype, device)
        std = torch.as_tensor(self.image_std, dtype, device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        """
        需要将图像限制在min_size和max_size之间，并对应缩放bboxes信息
        :param image:
        :param target:
        :return:
        """
        # [channel, height, width]
        h, w = image.shape[-2:]
        img_shape = torch.tensor(image.shape[-2:])
        img_min_size = float(torch.min(img_shape))  # 获取高宽中的最小值
        img_max_size = float(torch.max(img_shape))  # 获取高宽中的最大值

        if self.training:  # 如果处于训练过程中
            # 指定输入图片的最小边长[self.min_size]
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])

        # 根据指定最小边长与图片最小边长计算缩放比例
        scale_factor = size / img_min_size

        # 如果按照此缩放比例计算的图片最大边长大于指定的最大边长
        if img_max_size * scale_factor > self.max_size:
            # 则将缩放比例设置为 指定最大边长与图片最大边长之比
            scale_factor = self.max_size / img_max_size

        # 使用interpolate利用插值的方法缩放图片
        # image[None]是将图片从三维变成四维[None, channel, height, width]
        # 因为interpolate使用双线性插值方法，要求输入为 四维
        scaled_image = interpolate(image[None], scale_factor=scale_factor, mode="bilinear", align_corners=False)[0]

        # 如果target为空，则为验证模式，直接返回图片
        if target is None:
            return image, target

        # 如果target不为空，则为训练模式，需要按照同等缩放比例，对bbox进行缩放
        bbox = target["boxes"]

        # 根据图像的缩放比例来缩放bbox
        bbox = resize_boxes(bbox, (h, w), img_shape[-2:])
        target["bboxes"] = bbox

        return image, target

    def batch_images(self, images, size_divisible=32):
        """
        将一批图像打包成一个batch【便于后续计算】
        :param images: 输入的图片
        :param size_divisible: 将图像的高和宽调整到该数的整数倍
        :return:
        """

        # 计算一个batch中所有图片的最大channel、height、width
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)

        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)

        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [len(images)] - [batch]
        # maxsize - [channel, height, width]
        # batch_shape - [batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且全为0的Tensor
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes坐标不变
            pad_img[:images.shape[0], :images.shape[1], :images.shape[2]].copy_(img)

        return batched_imgs

    def max_by_axis(self, lst):
        # batch中第一张图片的shape[channel、height、width]
        maxes = lst[0]
        for sublist in lst[1:]:  # 从第二张图片开始
            for i, item in enumerate(sublist):  # 依次对比 channel、height、width
                maxes[i] = max(maxes[i], item)
        return maxes

    def forward(self, images, targets=None):
        """
        正向传播过程
        :param images:
        :param targets:
        :return:
        """
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors")

            image = self.normalize(image)  # 首先对图像进行标准化处理
            image, target = self.resize(image, target)  # 对图像和对应的标签缩放到指定范围

            images[i] = image

            if targets is not None and target is not None:
                targets[i] = target

        # 记录resize之后的图像尺寸
        image_sizes = [img.shape[-2:] for img in images]
        # 将images打包成一个batch
        batched_images = self.batch_images(images)

        image_size_list = list()
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_size_list.append((image_size[0], image_size[1]))

        # 将打包后的图片以及resize之后的尺寸
        image_list = ImageList(batched_images, image_size_list)

        # 返回数据【即将输入backbone】
        return image_list, targets

    def postprocess(self,
                    result,  # type: List[Dict[str, Tensor]]
                    image_shapes,  # type: List[Tuple[int, int]]
                    original_image_sizes  # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
        return result


def resize_boxes(bbox, original_size, expected_size):
    """
    根据同等缩放比例，对bbox进行缩放
    :param bbox:
    :param original_size:  原始尺寸
    :param expected_size:  期望尺寸
    :return:
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=bbox.device) /
        torch.tensor(s_origin, dtype=torch.float32, device=bbox.device)
        for s, s_origin in zip(expected_size, original_size)
    ]

    # 在高度、宽度上的缩放比例
    ratio_height, ratio_width = ratios

    # 将bbox在索引为1的维度展开，得到bbox信息【minibatch, 4】
    xmin, ymin, xmax, ymax = bbox.unbind(1)
    xmin *= ratio_width
    xmax *= ratio_width
    ymin *= ratio_height
    ymax *= ratio_height

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
