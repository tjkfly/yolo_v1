import os
import sys
from torch.utils import data
from PIL import Image, ImageDraw
import sys
import torch
import random
import cv2
import numpy as np
from data_utils.bounding_box import *
from data.transform import build_transform, build_target_transform

# from data.transform import TargetTransoform

# import build_target_transform, build_transfrom
from data_utils.utils import *


class VOCDatasets(data.Dataset):
    def __init__(self, data_txt, train=False, split=None, transform=None):
        self.train = train
        self.label_path = []
        self.image_path = []
        self.transform = transform
        # data_txt 每一行都是一个图片的路径，和一个图片label 的路径
        with open(data_txt) as f:
            lines = f.readlines()
        self.num_samples = len(lines)
        for line in lines:
            splited = line.strip().split(' ')
            self.image_path.append(splited[0])  # 图片的路径
            self.label_path.append(splited[1])  # label的路径
        # print(len(self.image_path), self.image_path[0])
        # print(len(self.label_path), self.label_path[0])
        # self.grid = Grid(True, True, rotate=1, offset=0, ratio=0.5, mode=1, prob=0.7)  # 这里只是初始化给参数  然后后面 干啥呀

    def _get_label(self, file, size):
        tmp = open(file, 'r')
        gt = []
        labels = []
        difficult = []
        for f in tmp.readlines():
            a = list(map(float, f.strip().split(',')))
            gt.append([a[0], a[1], a[2], a[3]])
            labels.append(int(a[4]))
        tmp.close()
        np.array(gt, dtype=np.float32)
        np.array(labels, dtype=np.int64)
        # np.array(difficult, dtype=np.uint8)
        return gt, labels  # 一个类 包含坐标位置，类别，是否困难样本

    def _data_aug(self, img, gt_list):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)  # cv2.flip(）图像翻转   1,0,-1   ,水平，垂直，水平垂直
            gt_list.flip(1)

        if random.random() > 0.5:
            img = torch.from_numpy(img) / 255.  # 图像归一化
            img = img.permute((2, 0, 1))  # 通道交换  BGR = > R G B

            # img, label = self.grid(img, gt_list)  # gt_list 在这里没有变化
            img = img.permute((1, 2, 0))
            img = img * 255
            img = img.numpy()
            img = img.astype(np.uint8)

        if random.random() > 0.2:
            img, gt_list = random_affine(img, gt_list, degrees=5, translate=.1, scale=.1, shear=2, border=0)

        if random.random() > 0.2:
            matrix = get_random_crop_tran(img)
            h, w, _ = img.shape
            img = cv2.warpAffine(img, matrix, (w, h))
            gt_list.warpAffine(matrix, (w, h))

        return img, gt_list

    def __getitem__(self, item):
        file_name = self.image_path[item]  # 图片路径
        gt_path = self.label_path[item]  # 标注文件路径 txt
        # print(file_name)
        # print(gt_path)
        # print(os.getcwd())
        file_name = os.path.join("D:\self\pytorch_learn\pytorch_proj\yolov1\yolov1_pytorch_tjk/", file_name)
        gt_path = os.path.join("D:\self\pytorch_learn\pytorch_proj\yolov1\yolov1_pytorch_tjk/", gt_path)

        # img = Image.open(file_name)  # 从路径中读取图片
        # img = cv2.imread(file_name)
        img = Image.open(file_name).convert("RGB")
        # print(file_name)
        temp_img = img

        img = np.array(img)
        # print(img.shape)
        # 读取标注文件
        boxes, labels = self._get_label(gt_path, (img.shape[1], img.shape[0]))
        # for box in boxes:
        #     draw = ImageDraw.Draw(temp_img)
        #     draw.rectangle([box[0], box[1], box[2], box[3]], outline='red')
        # Image._show(temp_img)

        # print(boxes, labels)
        # print(img.shape)
        # boxes : [x1, y1, x2, y2]
        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)  # 这里做的是数据集增强
            boxes = np.clip(boxes, 0.0, 1.0)
        # print(boxes)
        # if self.train:
        # 这里boxes 需要[xc,yc,w,h]
        #
        image, targets = build_target_transform(img, boxes, labels)  # 这里是要转换为合适的格式，便于计算 loss
        # return image, targets, labels

        # np.set_printoptions(threshold=100000)  # 全部输出
        # print(targets.shape)
        # print(image.shape)
        return image, targets  # 返回的是图像和真实的标注数据，

    #
    def __len__(self):
        return self.num_samples


## 加载voc数据集
def data_voc_loader(data_txt, train=False, image_size=448,
                    batch_size=4, num_workers=1):
    # print("data_voc_loader 加载数据集")
    dataset = VOCDatasets(data_txt, train,
                          split='train',
                          transform=build_transform('train', image_size)  # build_transfrom
                          )  # 构建 DataLoader 需要的 dataset

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    from data_utils.data_build import data_voc_loader

    path = 'D:\\self\\pytorch_learn\\pytorch_proj\\yolov1\\yolov1_pytorch_tjk\\train.txt'
    train_loader = data_voc_loader(path, image_size=448, batch_size=4, train=True)

    print(len(train_loader))
    for dd in train_loader:
        a, b = dd
        print(a.shape)
        print(b.shape)
        break

    # np.set_printoptions(threshold=100000)  # 全部输出
    # print(img.shape)
    # print(target)
