import math
import numpy as np
from data.augmentions import (Compose, ConvertColor, ConvertFromInts, Expand,
                              PhotometricDistort, RandomMirror,
                              RandomSampleCrop, Resize, SubtractMeans,
                              ToPercentCoords, ToTensor)


def build_transform(split, img_size):
    if split == 'train':
        transform = [ConvertFromInts(),
                     PhotometricDistort(),

                     # RandomSampleCrop(),
                     # RandomMirror(),  # 镜像
                     ToPercentCoords(),
                     Resize(img_size),

                     # SubtractMeans([123, 117, 104]),
                     ToTensor()
                     ]
    else:
        transform = [
            ConvertFromInts(),
            Resize(img_size),
            # SubtractMeans([123, 117, 104]),
            ToPercentCoords(),
            ToTensor()
        ]
    return Compose(transform)


def build_target_transform(img, boxes, labels):
    # 需要改为传参形式
    target_shape = (7, 7, 30)
    class_nums = 20
    cell_nums = 7
    #print("坐标转换")
    """
    传入的是一张图像上所含的gt box 数量 及对应的label数量
    :param img:
    :param boxes: boxes = [0.2 0.3 0.4 0.8]
    :param labels:labels = [1,2,3,4]
    :return:[self.S,self.S,self.B*5+self.C]
    """

    np_target = np.zeros(target_shape)
    # len(boxes) 为标注框的个数
    np_class = np.zeros((len(boxes), class_nums))
    for i in range(len(labels)):
        # n 行 20 列
        np_class[i][labels[i]] = 1
    step = 1.0 / cell_nums

    for i in range(len(boxes)):
        box = boxes[i]

        label = np_class[i]
        cx, cy, w, h = box
        # 归一化后的
        # 获取中心点所在的格子,3.5 实际是第四个格子，但是0为第一个，所以索引为3
        bx = int(cx // (step + 1e-5))
        by = int(cy // (step + 1e-5))


        cx = (cx % step) / step
        cy = (cy % step) / step

        box = [cx, cy, w, h]
        # print("box:",box)
        np_target[by][bx][:4] = box
        np_target[by][bx][4] = 1      # 这个格子有目标
        np_target[by][bx][5:9] = box
        np_target[by][bx][9] = 1
        np_target[by][bx][10:] = label  # 只预测一个类别
    return img, np_target
