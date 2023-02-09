from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks



class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x['type'] == 'convolutional'):
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernal_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernal_size - 1) //2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernal_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            if activation == "leaky":
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activ)
            
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
    
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor= 2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])  # the value of end can only be equal to or smaller than the index
            except:
                end = 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer() ## insert into the net as an empty layer directly
            module.add_module("route_{0}".format(index), route)
            if end< 0:  # there exists end
                filters = output_filters[index+start] + output_filters[index+end]
            else:       # no end existing
                filters = output_filters[index+start]
            
        ## shortcut == skip connection
        elif x["type"] == 'shortcut': 
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo -- detection layer
        elif x["type"] == "yolo":
            mask = x['mask'].split(',')
            mask = list(map(int, mask))

            anchors = x['anchors'].split(',')
            anchors = list(map(int, anchors))

            step = len(anchors) // len(mask)
            anchors = [tuple(anchors[i:i+1]) for i in range(0, len(anchors), 2)] ## divide anchors by step
            anchors = [anchors[i] for i in mask]  ## match anchors with mask

            detection = DetectionLayer(anchors)  # used to detect anchors of bbox
            module.add_module("Detection_{}".format(index), detection)
        
        # sum up
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)

# blocks = parse_cfg("/Users/gracelu/Documents/VsCodeP/pytorch/target detection/yolo/cfg/yolov3.cfg")
# print(blocks)

class Darknet(nn.Module):
    def __init__(self, cfgfile) -> None:
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        