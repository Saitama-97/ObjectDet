# _*_ coding: utf-8 _*_

"""
  @Time    : 2024/1/2 16:49 
  @File    : split_data.py
  @Project : Object_Det
  @Author  : Saitama
  @IDE     : PyCharm
  @Des     : 切分数据集【训练集:测试集 = 5:5】
"""

import os.path
import random

import os.path
import random

filepath = "../VOCdevkit/VOC2012/Annotations"

if not os.path.exists(filepath):
    print("文件夹不存在")
    exit(1)

val_rate = 0.5

files_name = sorted(file.split(".")[0] for file in os.listdir(filepath))

# 文件总数
files_num = len(files_name)

val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))

# 训练集及测试集 文件名
train_files = list()
val_files = list()

for i, file in enumerate(files_name):
    if i not in val_index:
        train_files.append(file)
    else:
        val_files.append(file)

try:
    train_txt = open("train.txt", "a")
    val_txt = open("val.txt", "a")
    train_txt.write("\n".join(train_files))
    val_txt.write("\n".join(val_files))
except FileExistsError as e:
    print(e)
    exit(1)
