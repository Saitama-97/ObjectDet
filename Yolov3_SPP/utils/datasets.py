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

import cv2
import numpy as np
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

# get orientation in exif tag
# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_wh(img_f):
    """
    获取图像的宽度和高度
    :param img_f: 图像路径
    :return:
    """
    img = Image.open(img_f)

    # wh - (width, height)
    wh = img.size

    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270  顺时针翻转90度
            wh = (wh[1], wh[0])
        elif rotation == 8:  # rotation 90  逆时针翻转90度
            wh = (wh[1], wh[0])
    except:
        pass

    return wh


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
            self.img_files = ['.' + file.strip() for file in lines]
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

        # 检查是否有shape文件（存储每张图像的宽高），如果没有则生成
        shape_file_path = path.replace(".txt", ".shapes")
        try:  # 已经有shape文件
            with open(shape_file_path, "r", encoding="utf-8") as rf:
                lines = list(rf.readlines())
                assert len(lines) == self.num_image, "shape file out of async"
                img_wh_list = [line.strip().split() for line in lines]
        except Exception as e:  # 没找到shape文件或者shape文件有错误，新建一份
            print("read {} failed, rebuild...".format(shape_file_path))
            img_wh_list = list()
            for img_f in tqdm(self.img_files, desc="Reading image shape"):
                # 获取图像的宽高信息
                img_wh = get_wh(img_f)
                img_wh_list.append(img_wh)
            np.savetxt(shape_file_path, img_wh_list, fmt="%g")
        # 每张图片的原始尺寸
        self.shapes = np.array(img_wh_list, dtype=np.float64)

        # Rectangular Training https://github.com/ultralytics/yolov3/issues/232
        # 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        # 注意: 开启rect后，mosaic就默认关闭
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            # 计算每个图片的高/宽比
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            # argsort函数返回的是数组值从小到大的索引值
            # 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
            irect = ar.argsort()
            # 根据排序后的顺序重新设置图像顺序、标签顺序以及shape顺序
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # set training image shapes
            # 计算每个batch采用的统一尺度
            shapes = [[1, 1]] * self.num_batch  # nb: number of batches
            for i in range(self.num_batch):
                ari = ar[bi == i]  # bi: batch index
                # 获取第i个batch中，最小和最大高宽比
                mini, maxi = ari.min(), ari.max()

                # 如果高/宽小于1(w > h)，将w设为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果高/宽大于1(w < h)，将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        # 缓存标签
        # 每张图片一个标签列表-[class, x, y, w, h]
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * self.num_image
        self.imgs = [None] * self.num_image
        extract_bounding_boxes, labels_loaded = False, False
        num_miss, num_found, num_empty, num_duplicate = 0, 0, 0, 0  # number mission, found, empty, duplicate
        # 这里分别命名是为了防止出现rect为False/True时混用导致计算的mAP错误
        # 当rect为True时会对self.images和self.labels进行从新排序
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".rect.npy"  # saved labels in *.npy file
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".norect.npy"

        if os.path.isfile(np_labels_path):
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == self.num_image:
                # 如果载入的缓存标签个数与当前计算的图像数目相同则认为是同一数据集，直接读缓存
                self.labels = x
                labels_loaded = True

        # 处理进度条只在第一个进程中显示【多GP->多进程->主进程rank为0】
        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)
        else:
            pbar = self.label_files

        # 遍历载入标签信息
        for i, file in enumerate(pbar):
            if labels_loaded:  # 如果存在缓存
                # 标注信息[class, x, y, w, h]
                cxywh_list = self.labels[i]
            else:  # 不存在缓存，从文件读取标签信息
                try:
                    with open(file, "r", encoding="utf-8") as rf:
                        cxywh_list = list()
                        for line in rf.readlines():
                            cxywh_list.append(line.strip().split())
                        # 标注信息[class, x, y, w, h]
                        cxywh_list = np.array(cxywh_list, dtype=np.float32)
                except Exception as e:
                    print("Read {} failed!!!".format(file))
                    num_miss += 1
                    continue

            # 如果标注信息有效（不为空）
            if cxywh_list.shape[0]:
                assert cxywh_list.shape[1] == 5, ".5 label columns: {}".format(file)
                assert (cxywh_list >= 0).all(), "negative labels: {}".format(file)
                assert (cxywh_list[:, 1:] <= 1).all(), "out of bounds coordinate labels: {}".format(file)

            if np.unique(cxywh_list, axis=0).shape[0] < cxywh_list.shape[0]:
                # 检车是否有重复行
                num_duplicate += 1

            if single_cls:
                cxywh_list[:, 0] = 0

            self.labels[i] = cxywh_list
            num_found += 1

            # Extract object detection boxes for a second stage classifier
            if extract_bounding_boxes:
                p = Path(self.img_files[i])
                img = cv2.imread(str(p))
                h, w = img.shape[:2]
                for j, x in enumerate(cxywh_list):
                    f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                    if not os.path.exists(Path(f).parent):
                        os.makedirs(Path(f).parent)  # make new output folder

                    # 将相对坐标转为绝对坐标
                    # b: x, y, w, h
                    b = x[1:] * [w, h, w, h]  # box
                    # 将宽和高设置为宽和高中的最大值
                    b[2:] = b[2:].max()  # rectangle to square
                    # 放大裁剪目标的宽高
                    b[2:] = b[2:] * 1.3 + 30  # pad
                    # 将坐标格式从 x,y,w,h -> xmin,ymin,xmax,ymax
                    b = xywh2xyxy(b.reshape(-1, 4)).revel().astype(np.int)

                    # 裁剪bbox坐标到图片内
                    b[[0, 2]] = np.clip[b[[0, 2]], 0, w]
                    b[[1, 3]] = np.clip[b[[1, 3]], 0, h]
                    assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
            else:
                num_empty += 1

            # 处理进度条只在第一个进程中显示
            if rank in [-1, 0]:
                # 更新进度条描述信息
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    num_found, num_miss, num_empty, num_duplicate, num_image)
        assert num_found > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep


if __name__ == '__main__':
    LoadImageAndLabels(path="../data/my_train_data.txt")
