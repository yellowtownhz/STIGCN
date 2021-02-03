
import torch
from torch import nn
import pdb
import random

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

class WeakTripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(WeakTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0][random.randint(0,n-5)])
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

class MyTripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(MyTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, inputs_local, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # Compute pairwise distance for local feature
        dist_local = torch.pow(inputs_local, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_local = dist_local + dist_local.t()
        dist_local.addmm_(1, -2, inputs_local, inputs_local.t())
        dist_local = dist_local.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap_part, dist_an_part = [], []
        dist_ap_ori, dist_an_ori = [], []
        for i in range(n):
            dist_ap_ori.append(dist[i][mask[i]].max())
            dist_an_ori.append(dist[i][mask[i] == 0].min())
            dist_ap_part.append(dist_local[i][torch.max(dist[i][mask[i]],0)[1]])
            dist_an_part.append(dist_local[i][torch.min(dist[i][mask[i]==0],0)[1]])
        dist_ap_part = torch.cat(dist_ap_part)
        dist_an_part = torch.cat(dist_an_part)
        dist_ap_ori = torch.cat(dist_ap_ori)
        dist_an_ori = torch.cat(dist_an_ori)
        # Compute ranking hinge loss
        y = dist_an_part.data.new()
        y.resize_as_(dist_an_part.data)
        y.fill_(1)
        y = Variable(y)
        loss_ori = self.ranking_loss(dist_an_ori, dist_ap_ori, y)
        prec_ori = (dist_an_ori.data > dist_ap_ori.data).sum() * 1. / y.size(0)
        loss_part = self.ranking_loss(dist_an_part, dist_ap_part, y)
        return loss_ori, prec_ori, loss_part
