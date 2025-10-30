from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


SRM_npy = np.load(os.path.join(os.path.dirname(__file__), 'SRM_Kernels.npy'))

class se_block(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(se_block, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel//ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=in_channel//ratio, out_features=in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs): 
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b,c])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view([b,c,1,1])
        outputs = x * inputs
        return outputs

class SRMConv2d(nn.Module):

    def __init__(self, stride=1, padding=0):
        super(SRMConv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        #padding=self.padding  1
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)



class BlockA(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=None):
        super(BlockA, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        self.SE=se_block(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out=self.SE(out)
        out += identity
        out = self.relu(out)

        return out


class BlockB(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=None):
        super(BlockB, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes, stride=2)

        self.bn1 = norm_layer(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.SE=se_block(out_planes)
        self.shortcut_conv = conv1x1(in_planes, out_planes, stride=2)
        
        self.shortcut_bn = norm_layer(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.pool(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
       
        out=self.SE(out)
        identity = self.shortcut_conv(identity)
        identity = self.shortcut_bn(identity)

        out += identity
        out = self.relu(out)

        return out
    


class Dilate_BlockA(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=None):
        super(Dilate_BlockA, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.dilate_conv1=nn.Conv2d(in_channels=in_planes ,out_channels=out_planes ,kernel_size = 3 , stride = 1,padding=1,dilation=1)  
        self.bn1=norm_layer(out_planes)
        # inplace-选择是否进行覆盖运算  
        self.relu=nn.ReLU(inplace=True) 
        self.dilate_conv2=nn.Conv2d(in_channels=out_planes ,out_channels=out_planes ,kernel_size = 3 , stride = 1,padding=2,dilation=2)
        self.bn2=norm_layer(out_planes)
    
        self.dilate_conv3=nn.Conv2d(in_channels=out_planes ,out_channels=out_planes ,kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.bn3=norm_layer(out_planes)
        self.dilate_conv4=nn.Conv2d(in_channels=out_planes ,out_channels=out_planes ,kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.bn4=norm_layer(out_planes)

        self.SE1=se_block(out_planes)
        self.SE2=se_block(out_planes)
        
    def forward(self, x):
        residual=x
        out = self.dilate_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dilate_conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dilate_conv3(out)
        out = self.bn3(out)
        out = self.SE2(out)
        out = self.relu(out)
        out = self.dilate_conv4(out)
        out = self.bn4(out)
        
        out += residual
        out = self.relu(out)
        
        return out