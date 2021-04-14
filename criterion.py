import torch
import torch.nn as nn
from pytorch_msssim import SSIM
import numpy as np


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.convX = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1), bias=False)
        x_gradient = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        y_gradient = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        self.convX.weight = nn.Parameter(torch.from_numpy(x_gradient).float().unsqueeze(0).unsqueeze(0))
        self.convY.weight = nn.Parameter(torch.from_numpy(y_gradient).float().unsqueeze(0).unsqueeze(0))
        for conv in [self.convY, self.convX]:
            for para in conv.parameters():
                para.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_x = self.convX(pred)
        pred_y = self.convY(pred)
        target_x = self.convX(target)
        target_y = self.convY(target)
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred_x, target_x) + l1_loss(pred_y, target_y)
        return loss


class DepthLoss(nn.Module):
    def __init__(self, weight):
        super(DepthLoss, self).__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        self.SSIM = SSIM()
        self.gradient_loss = GradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return self.weight * self.l1_loss(pred, target) + self.gradient_loss(pred, target) + (
                    1 - self.SSIM_loss(pred, target)) / 2
