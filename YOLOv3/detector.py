#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   detector.py
@Time    :   2020/12/31 11:46:16
@Author  :   lzneu
@Version :   1.0
@Contact :   lizhuang009@ke.com
@License :   (C)Copyright 2020-2021,KeOCR
@Desc    :   输入输出pipeline
'''

# here put the import lib
from __future__ import division
import time
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from util import *
import util
import os
import os.path as osp
import pickle as pkl
from darknet import DarkNet as DarkNet
import pandas as pd
import random

# 解析参数的方法
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLOv3 Detection Module') 
    parser.add_argument('--images', 
                        dest='images', 
                        help='Image / Directory contain images to perform detection upon',
                        default='imgs', 
                        type=str)
    parser.add_argument('--det', 
                        dest='det',
                        help='Image/Directory to store detections to',
                        default = "det", 
                        type = str)  # dest - 被添加到 parse_args() 所返回对象上的属性名
    parser.add_argument('--bs', 
                        dest='batch_size', 
                        help='Batch size',
                        default=1)
    parser.add_argument('--confidence',
                        dest='confidence',
                        help='Object Confidence to filter predictions',
                        default=0.6)
    parser.add_argument('--nms_thresh',
                        dest='nms_thresh',
                        help='NMS Threshhold',
                        default=0.4)
    parser.add_argument('--cfg',
                        dest='cfgfile',
                        help='Config file',
                        default='cfg/yolov3.cfg',
                        type=str)
    
    parser.add_argument("--weights", 
                        dest='weightsfile', 
                        help="weightsfile",
                        default="../weights/yolov3.weights", 
                        type=str)
    parser.add_argument("--reso", 
                        dest='reso', 
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", 
                        type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    images = args.images
    batch_size = int(args.batch_size)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    num_classes = 80    #For COCO
    classes = load_classes("../data/coco.names")
    classes[0] = 'beautiful girl'
    
    # 获取network 加载参数
    print("Loading network......")
    model = DarkNet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print('Nerwork successfully loaded')
    model.net_info['height'] = args.reso
    inp_dim = int(model.net_info['height'])
    assert inp_dim % 32 == 0
    assert inp_dim > 32
    if CUDA:
        model = model.cuda()
    model.eval() # 梯度不计算， drop out # 


    # 读取图片
    read_dir = time.time()
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    if not os.path.exists(args.det):
        os.makedirs(args.det)
    load_batch = time.time()
    loaded_ims = [cv2.imread(x) for x in imlist]  # BGR 
    # BGR -> RGB: B C H W
    im_batches = list(map(letterbox_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
     
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]  # N 2
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)  # N 4
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    # 创建Batch
    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat(im_batches[i*batch_size: min((i+1)*batch_size, len(im_batches))]) \
                        for i in range(num_batches)]
    
    # 开始检测
    write = 0
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        # 加载图片
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(batch, CUDA)
            output_recast = time.time()
            prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)
            end = time.time()
            if type(prediction) == int:
                # 说明没有预测结果
                for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
                    im_id = i * batch_size + im_num
                    print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", ""))
                    print("----------------------------------------------------------")
                continue
            
            # 将预测结果转化为im_list中的索引
            prediction[:,0] += i*batch_size 

            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat(output, prediction)
            
            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------------------------------------------------")
            if CUDA:
                torch.cuda.synchronize()   

            try:
                output
            except NameError:
                print ("No detections were made")
                exit()
            
        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        class_load = time.time()
        colors = pkl.load(open('pallete', 'rb'))
        
        draw = time.time()
        list(map(lambda x: util.write(x, loaded_ims, colors, classes), output))
        det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
        # 写入文件
        list(map(cv2.imwrite, det_names, loaded_ims))
        end = time.time()
            
        print("SUMMARY")
        print("----------------------------------------------------------")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print()
        print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
        print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
        print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
        print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
        print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
        print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
        print("----------------------------------------------------------")