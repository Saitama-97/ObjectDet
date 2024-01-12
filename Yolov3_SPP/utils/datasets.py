# _*_ coding: utf-8 _*_

"""
  @Time    : 2024.01.12 10:45
  @File    : datasets.py
  @Project : ObjectDet
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 自定义数据集
"""
import os.path
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class LoadImageAndLabels(Dataset):
    def __init__(self,
                 path,  # 指向my_train_data.txt & my_val_data.txt，训练集及验证集中每一张图片的路径
                 img_size=416,  # 训练时，表示多尺度训练中的最大尺寸；验证时，表示最终使用的网络大小
                 batch_size=16,  # 取决于硬件
                 augment=False,  # 是否开启图像增强（训练开/验证关）
                 hyp=None,  # 超参数字典，包含图像增强过程中使用的超参数
                 rect=None,  # 是否使用rectangular training（训练时不开启，验证时开启）
                 cache_images=False,  # 是否将图片缓存到内存中
                 single_cls=False,
                 pad=0.0,
                 rank=-1):  # 使用单GPU默认为-1，使用多GPU则会开启多进程，主进程rank=0
        try:
            path = str(Path(path))
            if os.path.isfile(path):
                with open(path, "r") as rf:
                    lines = rf.readlines()
            else:
                raise Exception("{} doesn't exist".format(path))
            self.img_files = [file.strip() for file in lines]
            print()
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}".format(path, e))

        # 图片个数
        num_image = len(self.img_files)
        assert num_image > 0, "No images found in {}".format(path)

        # 根据传入的batch给图片分批
        bi = np.floor(np.arange(num_image) / batch_size).astype(np.int)
        # 总batch数
        num_batch = bi[-1] + 1

        self.num_image = num_image  # 图片总数
        self.num_batch = num_batch  # 总batch数
        self.batch = bi  # 哪张图片属于哪个batch [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 ...]
        self.img_size = img_size  # 预处理之后图片的输出尺寸
        self.aug = augment  # 是否开启图像增强（训练开/验证关）
        self.hyp = hyp  # 超参数字典，包含图像增强过程中使用的超参数
        self.rect = rect  # 是否使用rectangular training（训练时不开启，验证时开启）
        self.mosaic = self.aug and not self.rect

        # 所有标签文件路径
        self.label_files = [file.replace("images", "labels").replace(".jpg", ".txt") for file in self.img_files]

        # 检查是否有shape文件（存储每张图像的宽高），如果没有则升成

        print()


if __name__ == '__main__':
    LoadImageAndLabels(path="../data/my_train_data.txt")
