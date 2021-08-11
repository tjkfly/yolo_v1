import torch
from torch import nn


class YoloLoss(nn.Module):
    def __init__(self, num_class=20):
        super(YoloLoss, self).__init__()
        self.lambda_coord = 5  # 论文loss 中的参数
        self.lambda_noobj = 0.5
        self.S = 7
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 7

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        box = torch.clamp(box, 0)
        return box

    def compute_iou(self, box1, box2, index):
        box1 = torch.clone(box1)
        box2 = torch.clone(box2)
        # 坐标转换
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        # 获取相交
        inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
        inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))
        inter_h = torch.clamp(inter_h, 0)
        inter_w = torch.clamp(inter_w, 0)
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return inter / union

    def forward(self, pred, target):
        batch_size = pred.size(0)
        """
        contiguous：view只能用在contiguous的variable上。
        如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        一种可能的解释是：
        有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        判断是否contiguous用torch.Tensor.is_contiguous()函数。
        """
        # 关于坐标
        target_boxes = target[:, :, :, :10].contiguous().reshape((-1, 7, 7, 2, 5))
        pred_boxes = pred[:, :, :, 10:].contiguous().reshape((-1, 7, 7, 2, 5))
        # 关于类别
        target_class = target[:, :, :, 10:]
        pred_class = pred[:, :, :, 10:]

        obj_mask = (target_boxes[..., 4] > 4).byte()  # [b, 7, 7, 2]
        sig_mask = obj_mask[..., 1].bool()  # [b, 7, 7]
        index = torch.where(sig_mask == True)

        for img_i, y, x in zip(*index):
            img_i, y, x = img_i.item(), y.item(), x.item()
            pbox = pred_boxes[img_i, y, x]
            target_box = target_boxes[img_i, y, x]
            ious = self.compute_iou(pbox[:, :4], target_box[:, :4], [x, y])
            iou, max_i = ious.max(0)
            pred_boxes[img_i, y, x, max_i, 4] = iou.item()    # 最大的等于 iou
            pred_boxes[img_i, y, x, 1 - max_i, 4] = 0         # 否则等于 0
            obj_mask[img_i, y, x, 1 - max_i] = 0

        obj_mask = obj_mask.bool()
        noobj_mask = ~obj_mask()

if __name__ == '__main__':
    target_boxes = torch.tensor([[
        [[[1, 2, 3, 4, 0],
          [1, 2.1, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 1],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]]],
        [[[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 1],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]]],
        [[[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 1],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]]],
        [[[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 1],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]]],
        [[[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 1],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]]],
        [[[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 1],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]]],
        [[[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 0]],
         [[1, 2, 3, 4, 1],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]],
         [[1, 2, 3, 4, 0],
          [1, 2, 3, 4, 1]]]
    ]])

print(target_boxes.shape)
print(target_boxes[..., 4].shape)
a = (target_boxes[..., 4] > 0).byte()
# print(a.shape)
print(a)

sig_mask = a[..., 1].bool()
b = a[..., 0].bool()

index = torch.where(sig_mask == True)
print(sig_mask.shape)
print(sig_mask)
print(index)
