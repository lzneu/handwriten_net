#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2020/12/29 20:26:06
@Author  :   lzneu
@Version :   1.0
@Contact :   lizhuang009@ke.com
@License :   (C)Copyright 2020-2021,KeOCR
@Desc    :   
'''

from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    接收网络的输出特征，转化为二维的bbox预测
    归一化多尺度的输出，转化为统一的表示形式
    :param: inp_dim: 输入图片的尺寸
    :return
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    # (N, bbox_attrs * 3, h, w)
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    # 将anchor的尺寸转化为该特征图的尺寸
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # 将tx, ty进行sigmoid
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # 增加在grid 上的offset
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)  
    x_offset = torch.FloatTensor(a).view(-1, 1)  # [0, 1, 2, ..., 13, 0, 1, ...]
    y_offset = torch.FloatTensor(b).view(-1, 1)  # [0, 0, 0, ..., 13, 13]
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    # 数量为13*13*3
    x_y_offset = torch.cat((x_offset, y_offset), dim=1).repeat(1, num_anchors).view(-1 ,2).unsqueeze(0)
    # bx = sigmoid(tx) + cx
    prediction[:, :, :2] += x_y_offset

    # 应用anchor
    anchors = torch.FloatTensor(anchors)  # (3, 2)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)  # (1, 13*13*3, 2)
    # bw = pw * exp(tw)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # (N, 13*13*3, 2)

    # class score
    prediction[:, :, 5: 5+num_classes] = torch.sigmoid(prediction[:, :, 5: 5+num_classes])

    # 最后一步，将bbox回到输入图片的大小
    prediction[:, :, :4] *= stride
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    objectness 置信度过滤、NMS
    :param: prediction(N, 10674, 5+80)
    :return
    """
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)  # N 10674 1
    # 低于阈值的pred 全部置为0
    prediction = prediction * conf_mask  # (N, 10674, 5+80) * (N, 10674, 1) = (N, 10674, 5+80) 这里进行了广播
    
    # 转化为左上右下两个坐标
    box_corner = prediction.new(prediction.shape)  # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2]/2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3]/2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2]/2 
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3]/2 
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)
    write = False
    for ind in range(batch_size):
        image_pred = prediction[ind]  # (10674, 85)
        # NMS
        # 由于只关心80个类中的最大的分数，因此将80个元素转化为max及其class_ind
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+ num_classes], dim=1)
        max_conf = max_conf.float().unsqueeze(1)  # (10674, 1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, dim=1)
        # 过滤低于阈值的预测
        non_zero_ind = torch.nonzero(image_pred[:, 4])  # 10674
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)  # N 7
        except:
            continue
        # 没有预测结果大于阈值的情况下，
        if image_pred_.shape[0] == 0:
            continue
        # 获得该图片出现的类别索引
        img_classes = unique(image_pred_[:, -1])  # 10674 -> 类别数量  
        for cls in img_classes:            
            # 获取该类别的image_pred_
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)  # 应用pytorch的广播，N 7 
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()  # N
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)  # N 7
            # objectness置信度排序 
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # 检测结果的数量
            
            # 执行nms
            for i in range(idx):
                # 计算iou
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])  # (i+1:)
                except ValueError:
                    break
                except IndexError:
                    break
                # 过滤大于阈值的detections
                iou_mask = (ious < nms_conf).float().unsqueeze(1)  # (i+1:, 1)
                image_pred_class[i+1: ] *= iou_mask  # 利用pytorch的广播机制
                # 移除
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)  #  D 7

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)   
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq, dim=1)
                    write = True
                else:
                    out = torch.cat(seq, dim=1)
                    output = torch.cat((output, out))
    try:
        return output
    except:
        return 0  # 说明output未初始化，无detectioins



def bbox_iou(box1, box2):
    """
    计算两个iou
    :param box1: (n1, 7)
    :param box2: (n2, 7)
    :return ious: (n1, n2)
    """
    # 获取坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # 获取相交面积
    inter_rect_x1 = torch.max(b1_x1, b2_x1) # n1, n2
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_rect_area = torch.clamp(inter_rect_x2-inter_rect_x1+1, min=0) * torch.clamp(inter_rect_y2-inter_rect_y1+1, min=0)   # n1, n2
    # 获取union面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    # iou
    iou = inter_rect_area / (b1_area + b2_area - inter_rect_area) 
    return iou

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def load_classes(namesfile):
    """加载类别数量"""
    with open(namesfile, 'r') as fp:
        names = fp.read().split('\n')[: -1]
        return names


def prep_image(img, inp_dim):
    """
    将cv2的BGR图片转化为torch的 RGB BCWH格式
    :param
    :return tensor
    """
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1]  # BGR->RGB
    img = img.transpose((2, 0, 1)).copy()  # HWC-> CHW
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)  # BCHW
    return img




def letterbox_image(img, inp_dim):
    """
    resize图片，不够的地方用padding，填充值(128, 128, 128)
    :param
    :return
    """
    inp_dim = (inp_dim, inp_dim)
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    ratio = min(w/img_w, h/img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2: (h-new_h)//2 + new_h, (w-new_w)//2: (w-new_w)//2+new_w, :] = resized_image  # HWC
    canvas = canvas.transpose((2, 0, 1)).copy()  # HWC-> CHW
    canvas = torch.from_numpy(canvas).float().div(255.0).unsqueeze(0)  # BCHW
    return canvas




if __name__ == "__main__":
    # 增加在grid 上的offset
    a = torch.arange(6).reshape((2, 3))
    b = a.unsqueeze(1).contiguous()  # 2, 1, 3
    print(a.size(), b.size())
    print(a)
    print(b)
    c = a * b 
    print(c.size())
    print(c)


    CUDA = False
    grid_size = 13
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)  
    x_offset = torch.FloatTensor(a).view(-1, 1)  # [0, 1, 2, ..., 13, 0, 1, ...]
    y_offset = torch.FloatTensor(b).view(-1, 1)  # [0, 0, 0, ..., 13, 13]
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), dim=1).repeat(1, 3).view(-1 ,2).unsqueeze(0)
    print(x_y_offset)

def write(x, results, colors, classes):
    c1 = tuple(x[1:3].int())  # 左上
    c2 = tuple(x[3:5].int())  # 右下
    img = results[int(x[0])]
    cls = int(x[-1])
    color = colors[cls]
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img