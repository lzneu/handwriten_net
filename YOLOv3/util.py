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

def write_results(prediction, confidence, num_classes, num_conf=0.4):
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
        max_conf = max_conf.float().unsqeeze(1)  # (10674, 1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, dim=1)
        # 过滤低于阈值的预测
        non_zero_ind = torch.nonzero(image_pred[:, 4])  # 10674
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue
        # 没有预测结果大于阈值的情况下，
        if image_pred_.shape[0] == 0:
            continue
        # 获得该图片出现的类别索引
        img_classes = unique(image_pred_[:, -1])  # 10674 -> 类别数量
        for cls in img_classes:
            # 执行nms




def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res










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
