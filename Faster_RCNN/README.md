# Faster R-CNN

## 文件结构：

```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 数据集下载地址
- PascalVOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

## 预训练权重下载地址（下载后放入backbone文件夹中）：

* MobileNetV2 weights(下载后重命名为`mobilenet_v2.pth`，然后放到`bakcbone`
  文件夹下): https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN weights: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是`fasterrcnn_resnet50_fpn_coco.pth`文件，
  不是`fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`，然后放到当前项目根目录下即可。
 