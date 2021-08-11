import torch
import os
import sys
from model.yolo import build_yolo
from data_utils.data_build import data_voc_loader
from model.loss import YoloLoss
from utils.lr_scheduler import WarmupMultiStepLR

epochs = 1
img_size = 448
bs = 2
train_root = "D:\\self\\pytorch_learn\\pytorch_proj\\yolov1\\yolov1_pytorch_tjk\\train.txt"


def train():
    # train.txt 中包含txt标注文件的路径
    # 1、加载数据集
    train_loader = data_voc_loader(train_root, image_size=448, batch_size=4, train=True)
    # # 2、模型相关
    model = build_yolo().cuda()
    # model = build_yolo()
    lr = 1e-3
    optim = torch.optim.SGD(model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=5e-4)
    milestones = None
    lr_scheduler = WarmupMultiStepLR(
        optimizer=optim,
        milestones=[80000, 100000] if milestones is None else milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500)
    # state_dict = torch.load()
    # model.load_state_dict(state_dict)
    # print(model)
    # # criterion = yoloLoss().cuda()
    criterion = YoloLoss().cuda()
    # # # img, target = Variable(img).cuda(), Variable(target).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        for step, train_data in enumerate(train_loader):
            img, target = train_data
            img = img.to(device)           # GPU
            target = target.to(device)     # GPU
            # print(img.shape)
            # print(target.shape)
            output = model(img)
            optim.zero_grad()

            loss_dict = criterion(output, target.float())
            loss = sum(x for x in loss_dict.values())
            loss.backward()
            optim.step()
            lr_scheduler.step()

            # print(output)
            # print(output.shape)
            print(epoch, step)

            print(loss)
            # break

if __name__ == '__main__':
    train()
