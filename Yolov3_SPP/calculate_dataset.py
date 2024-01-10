# _*_ coding: utf-8 _*_

"""
  @Time    : 2024.01.10 16:09
  @File    : calculate_dataset.py
  @Project : ObjectDet
  @Author  : Saitama
  @IDE     : PyCharm
  @Desc    : 生成my_train_data.txt等文件
"""
import os

from tqdm import tqdm


def create_data_txt(txt_path, dataset_path):
    """
    根据已经分好的数据集(label)，生成训练集、测试集txt，记录每张图片的路径(确保图片存在)
    :param txt_path:
    :param dataset_path:
    :return:
    """
    with open(txt_path, "w") as wf:
        for file_name in tqdm(os.listdir(dataset_path), desc="Processing {}".format(txt_path)):
            # Yolov3_SPP/my_yolo_dataset/train/labels/2008_000008.txt
            image_path = os.path.join(dataset_path.replace("labels", "images"), file_name.replace(".txt", ".jpg"))

            assert os.path.exists(image_path), "file:{} doesn't exist".format(file_name)

            wf.write(image_path + "\n")


def create_data_data(output_path, label_num, train_path, val_path):
    """
    生成data.data文件，包含如下信息
    classes = 类别总数
    train = train_data.txt路径
    val = val_data.txt路径
    names = label.names路径
    :param output_path:
    :param label_path:
    :param train_path:
    :param val_path:
    :return:
    """

    with open(output_path, "w") as wf:
        wf.write("classes={}\n".format(label_num))
        wf.write("train={}\n".format(train_path))
        wf.write("val={}\n".format(val_path))
        wf.write("names=data/my_data_label.names")


def create_my_cfg_file(raw_cfg_path, output_path, label_num):
    """
    基于Yolo官方cfg和类别总数，生成自己的cfg
    :param raw_cfg_path: YOLO v3 官方cfg路径
    :param output_path: 自定义cfg路径
    :param label_num: 类别总数
    :return:
    """
    assert os.path.exists(raw_cfg_path), "raw cfg doesn't exist"

    filters_lines = [636, 722, 809]
    classes_lines = [643, 729, 816]

    cfg_lines = open(raw_cfg_path, "r").readlines()

    for i in filters_lines:
        assert "filters" in cfg_lines[i - 1], "filters param is not in line:{}".format(i - 1)
        output_num = (5 + label_num) * 3
        cfg_lines[i - 1] = "filters={}\n".format(output_num)

    for i in classes_lines:
        assert "classes" in cfg_lines[i - 1], "classes param is not in line:{}".format(i - 1)
        cfg_lines[i - 1] = "classes={}\n".format(label_num)

    with open(output_path, "w") as w:
        w.writelines(cfg_lines)


if __name__ == '__main__':
    train_txt_path = "data/my_train_data.txt"
    val_txt_path = "data/my_val_data.txt"
    label_path = "data/my_data_label.names"

    create_data_txt(train_txt_path, "./my_yolo_dataset/train/labels")
    create_data_txt(val_txt_path, "./my_yolo_dataset/val/labels")

    # 类别总数
    label_num = 0
    with open(label_path, "r") as rf:
        for line in rf.readlines():
            if line:
                label_num += 1

    create_data_data("./data/my_data.data",
                     label_num,
                     train_txt_path,
                     val_txt_path)

    # yolov3 官方cfg路径
    yolo_official_cfg_path = "cfg/yolov3-spp.cfg"
    # 自定义cfg路径
    my_cfg = "cfg/my_yolov3.cfg"
    create_my_cfg_file(yolo_official_cfg_path, my_cfg, label_num)
