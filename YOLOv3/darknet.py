#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   darknet.py
@Time    :   2020/12/29 14:55:08
@Author  :   lzneu
@Version :   1.0
@Contact :   lizhuang009@ke.com
@License :   (C)Copyright 2020-2021,KeOCR
@Desc    :   yolov3 backbone
'''

# here put the import lib
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pprint
from util import predict_transform
import cv2



class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        """
        占位层
        """
        super(EmptyLayer, self).__init__()

def parse_cfg(cfgfile):
    """
    解析配置文件 作为输入    
    :param
    :return: 返回一个blocks的list，将所有的变量存储到dict，构成的block
    """
    file_ = open(cfgfile, 'r')
    lines = file_.read().split('\n')
    file_.close()
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.lstrip().rstrip() for x in lines]

    # 转存成dict
    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":  # 说明是一个block的开始
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

def create_modules(blocks):
    """
    根据配置文件的blocks生成网络
    :param
    :return
    """
    net_info = blocks[0]  # net_info存储的是整个网络的信息，输入尺寸等
    module_list = nn.ModuleList()   # 存储网络
    prev_filters = 3  # 记录上一个层的输出通道数，用来搭建网络
    output_filters = []  # 记录每个层的输出通道数，配合route层使用

    # 开始搭建
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # 共有5种类型的模块

        #  卷积模块
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:  # 有的卷积层不带batch_normalize
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            if padding:
                pad = (kernel_size -1 ) // 2
            else:
                pad = 0
            # 添加卷积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size=kernel_size, padding=pad, stride=stride, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)
            # 添加bn层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)
            # 激活层
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)

        # 上采样层
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample)

        # Route Layer 主要用来concate输出特征
        elif (x["type"] == "route"):
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0].strip())
            try:
                end = int(x['layers'][1].strip())
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()  # 使用占位层，因为操作简单，不再定义层，后面直接在darknet中添加操作即可
            module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index+start] + output_filters[index+end]  # 这里进行了concate操作
            else:
                filters = output_filters[index+start]

        # Shortcut Layers
        elif x['type'] == 'shortcut':
            # pass 
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)
        
        # 最终 定义YOLO模块
        elif x['type'] == 'yolo':
            mask = list(map(int, x["mask"].split(",")))
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)

class DarkNet(nn.Module):
    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:] # 而且层的顺序与cfg中定义的相同
        outputs = {}  # 存储route layer的输出 k: 层的索引, v: 层的输出特征

        write = 0  # 用来标志是不是第一个尺度的特征，如果不是直接concate输出结果即可

        for i, module in enumerate(modules):
            module_type = module['type']
            
            # 卷积层或者上采样层，直接forward即可
            if module_type in ['convolutional', 'upsample']:
                x = self.module_list[i](x)  
            
            #  route层, concate操作
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i+layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), dim=1)  # 在通道维度上concate
            
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]
            
            # 在该尺度输出
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module["classes"])
                # 开始转化
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x 
                    write = 1
                else:
                    detections = torch.cat((detections, x), dim=1)
                
            # 把推理后的x赋值到outputs
            outputs[i] = x
        return detections
    
    def load_weights(self, weightfile):
        """
        加载参数，注意: 
            - 只存储了conv和bn层的参数
            - 按照cfg中的顺序存储
            - 带bn层的convolutional的卷积层不带bias
            - 不带bn层的convolutional的卷积层带bias
        :param
        :return
        """
        # 开始的160个字节存储5个int32 构成了文件头
        fp = open(weightfile, 'rb')
        # 1 Major version num
        # 2 Minor version num
        # 3 Subversion num
        # 4 5 Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # 剩余的为float32的weights
        weights = np.fromfile(fp, dtype=np.float32)

        # 加入到network中
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']
            # 如果 module_type是 convolutional 则加载weights 否则 continue
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                # conv + bn + activation
                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    # 获取bn层的bias的数量
                    num_bn_biases = bn.bias.numel()
                    # 加载weights
                    bn_biases = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    # 加入到模型中
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                # 如果不存在bn层，直接load 卷积层bias即可
                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr+num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                # 加载convolutional的weights
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr+num_weights])
                ptr += num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = torch.tensor(img_)                     # Convert to Variable
    return img_


if __name__ == "__main__":
    # 测试一下代码
    device = torch.device('cuda')
    inp = get_test_input()
    model = DarkNet(cfgfile='./cfg/yolov3.cfg')
    model.load_weights('../weights/yolov3.weights')
    model = model.to(device)
    print(inp)
    pred = model(inp.to(device), True)
    print(pred.size())
    print(pred)
