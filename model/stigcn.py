import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class unit_tcn(nn.Module):
    r""" No ReLU """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # input: NxM, C, T, V
        x = self.bn(self.conv(x))
        return x


class CovGraph(nn.Module):
    def __init__(self, in_channels, proj_channels):
        super().__init__()
        self.proj_channels = proj_channels
        self.proj1 = nn.Conv2d(in_channels, proj_channels, 1)
        self.proj2 = nn.Conv2d(in_channels, proj_channels, 1)
        self.soft = nn.Softmax(-2)
        self.reset_parameters()

    def reset_parameters(self):
        conv_init(self.proj1)
        conv_init(self.proj2)

    def forward(self, x):
        N, C, T, V = x.size()
        embed1 = self.proj1(x).permute(0, 3, 1, 2).contiguous().view(N, V,
                                                                     self.proj_channels * T)
        embed2 = self.proj2(x).view(N, self.proj_channels * T, V)
        A = self.soft(torch.matmul(embed1, embed2) / embed1.size(-1))  # N V V
        return A



class BlockS(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        """ adjacent matrix """
        A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        # joint
        self.A1 = A[0]
        self.PA1 = nn.Parameter(self.A1.clone(), requires_grad=True)
        nn.init.constant_(self.PA1, 1e-6)
        # bone
        self.A2 = A[0] - A[1]
        self.PA2 = nn.Parameter(self.A2.clone(), requires_grad=True)
        nn.init.constant_(self.PA2, 1e-6)
        # cheby
        A3 = A[1] + A[2]
        self.A3 = 4*torch.pow(A3, 2) - A3 - 2*torch.eye(A3.size(-1))
        self.PA3 = nn.Parameter(self.A3.clone(), requires_grad=True)
        nn.init.constant_(self.PA3, 1e-6)

        soft = nn.Softmax(-2)
        self.A4 = soft((8*torch.pow(A3, 4)- 4*torch.pow(A3, 2)-4*A3 +torch.eye(A3.size(-1)))/A3.size(-1))
        self.PA4 = nn.Parameter(self.A4.clone(), requires_grad=True)
        nn.init.constant_(self.PA4, 1e-6)

        """ branches """
        branch_channels = out_channels // 4
        # covariance dependency
        self.CA = CovGraph(in_channels, branch_channels)
        self.PCA = nn.Parameter(torch.ones(1))
        nn.init.constant_(self.PCA, 1e-6)

        self.branches = nn.ModuleList(
            [BasicConv2d(in_channels, branch_channels, kernel_size=1) for _ in range(4)])

        # fusion
        self.fusion = unit_tcn(4 * branch_channels, out_channels, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        CA = self.CA(x)

        # joint
        A1 = self.A1.to(x.device) + self.PCA * CA + self.PA1
        xj = x.view(N, C * T, V)
        xj = torch.matmul(xj, A1).view(N, C, T, V)
        xj = self.branches[0](xj)

        # bone
        A2 = self.A2.to(x.device) + self.PCA * CA + self.PA2
        xb = x.view(N, C * T, V)
        xb = torch.matmul(xb, A2).view(N, C, T, V)
        xb = self.branches[1](xb)

        # cheby
        A3 = self.A3.to(x.device) + self.PCA * CA + self.PA3
        xc = x.view(N, C * T, V)
        xc = torch.matmul(xc, A3).view(N, C, T, V)
        xc = self.branches[2](xc)

        A4 = self.A4.to(x.device) + self.PCA * CA + self.PA4
        xh = x.view(N, C * T, V)
        xh = torch.matmul(xh, A4).view(N, C, T, V)
        xh = self.branches[3](xh)

        out = torch.cat([xj, xb, xc, xh], dim=1)
        out = self.fusion(out)
        return out


class BlockT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 2
        self.branches = nn.ModuleList([
            unit_tcn(in_channels, branch_channels, 3) for _ in range(2)])
        self.relu = nn.ReLU(inplace=True)
        self.fusion = unit_tcn(2 * branch_channels, out_channels, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        out = self.relu(self.branches[0](x))

        # motion
        frame = list(range(1, T))
        frame.append(T - 1)
        xm = x[:, :, frame] - x
        xm = self.relu(self.branches[1](xm))

        out = torch.cat([out, xm], dim=1)
        out = self.fusion(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        self.spatial = BlockS(in_channels, out_channels, A)
        self.temporal = BlockT(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

        # shortcut
        if not residual:
            self.shortcut = lambda x: 0
        elif in_channels == out_channels:
            self.shortcut = lambda x: x
        else:
            self.shortcut = unit_tcn(in_channels, out_channels, kernel_size=1)

        if residual:
            bn_init(self.spatial.fusion.bn, 1e-6)
            bn_init(self.temporal.fusion.bn, 1e-6)

    def forward(self, x):
        out = self.spatial(x) + self.temporal(x)
        out += self.shortcut(x)
        return self.relu(out)


class DataBN(nn.Module):
    def __init__(self, num_person, in_channels, num_point):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.reset_parameters()

    def reset_parameters(self):
        bn_init(self.bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        return x


class Model(nn.Module):
    def __init__(self, layers=[3, 3, 3], num_class=60, num_point=25,
                 num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 dropout=0.5):
        super().__init__()
        # Graph
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.A = self.graph.A

        # Data process
        self.data_bn = DataBN(num_person, in_channels, num_point)

        # Blocks
        self.layer0 = BasicBlock(in_channels, 64, A, residual=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)

        self.inplanes = 64
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])

        self.layer2 = self._make_layer(BasicBlock, 128, layers[1])

        self.layer3 = self._make_layer(BasicBlock, 256, layers[2])

        # fc
        self.dropout = nn.Dropout(p=dropout)
        self.num_class = num_class
        self.fc = nn.Linear(self.inplanes, num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_class))
        nn.init.constant_(self.fc.bias, 0)

    def _make_layer(self, block, planes, num_blocks, **kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.inplanes, planes, self.A, **kwargs))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, T, V, M = x.size()
        # data bn
        x = self.data_bn(x)

        x = self.maxpool(self.layer0(x))
        x = self.maxpool(self.layer1(x))
        x = self.maxpool(self.layer2(x))
        x = self.layer3(x)

        # N*M,C,T,V
        x = x.view(N, M, x.size(1), -1)
        x = x.mean(3).mean(1)
        out = self.fc(self.dropout(x))

        return out
