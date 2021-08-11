import torch
import os
import sys
from model.yolo import build_yolo
from data_utils.data_build import data_voc_loader


epochs = 1
img_size = 448
bs = 2
train_root = "D:\\self\\pytorch_learn\\pytorch_proj\\yolov1\\yolov1_pytorch_tjk\\train.txt"


def train():
    # train.txt 中包含txt标注文件的路径
    # 1、加载数据集
    train_loader = data_voc_loader(train_root, image_size=448, batch_size=4, train=True)
    # # 2、模型相关
    # model = build_yolo().cuda()
    model = build_yolo()
    # print(model)
    # # criterion = yoloLoss().cuda()
    # # # img, target = Variable(img).cuda(), Variable(target).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        for step, train_data in enumerate(train_loader):
            img, target = train_data
            # img = img.to(device)           # GPU
            # target = target.to(device)     # GPU
            # print(img.shape)
            # print(target.shape)
            output = model(img)
            print(output)
            print(output.shape)
            break

if __name__ == '__main__':
    train()
