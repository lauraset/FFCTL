import torch
import torch.nn as nn
import torch.nn.functional as F
from losses_pytorch.iou_loss import IOU
from losses_pytorch.ssim_loss import SSIM


class CE_MSE(nn.Module):
    def __init__(self, weight=None, beta=0.7):
        super().__init__()
        self.weight = weight
        self.beta = beta # the balance term
    def forward(self, pmask, rmask, pbd,  rbd):
        ce = F.cross_entropy(pmask, rmask, weight=self.weight)
        mse = F.mse_loss(pbd, rbd.float()/255.) # normed to 0-1
        loss = ce + self.beta*mse
        return loss


class BCE_SSIM_IOU(nn.Module):
    def __init__(self, issigmoid = False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        self.ssim = SSIM(window_size=11, size_average=True)
        self.iou = IOU(size_average=True)
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask = torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_ssim + loss_iou
        return loss


# 2021.11.26:
class BCE_IOU(nn.Module):
    def __init__(self, issigmoid = False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        # self.ssim = SSIM(window_size=11, size_average=True)
        self.iou = IOU(size_average=True)
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask = torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        # loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss = loss_ce  + loss_iou
        return loss


class BCE_SSIM(nn.Module):
    def __init__(self, issigmoid = False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        self.ssim = SSIM(window_size=11, size_average=True)
        # self.iou = IOU(size_average=True)
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask = torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        loss_ssim = 1-self.ssim(pmask, rmask)
        # loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_ssim # + loss_iou
        return loss


class BCE_SSIM_IOU_BD(nn.Module):
    def __init__(self,issigmoid=False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        self.ssim = SSIM(window_size=11, size_average=True)
        self.iou = IOU(size_average=True)
        self.bd = torch.nn.MSELoss()
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask, pbd, rbd):
        if self.issigmoid:
            pmask=torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss_bd = self.bd(pbd, rbd.float()/255.)
        loss = loss_ce + loss_ssim + loss_iou + loss_bd
        return loss

# 2022.1.8
class Dice(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        # pred: sigmoid
        # targer: 0, 1
        smooth = 1.
        m1 = pred.view(-1)  # Flatten
        m2 = target.view(-1)  # Flatten
        intersection = (m1 * m2).sum()
        return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# 2022.1.8
# pmask: sigmoid
# rmask: 0,1
class BCE_DICE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        # self.ssim = SSIM(window_size=11, size_average=True)
        self.dice = Dice()
    def forward(self, pmask, rmask):
        loss_ce = self.ce(pmask, rmask)
        loss_dice = self.dice(pmask, rmask)
        loss = loss_ce  + loss_dice
        return loss

# 2022.4.18: add probability
class BCE_DICE_Prob(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='none')
        # self.ssim = SSIM(window_size=11, size_average=True)
        self.dice = Dice()
    def forward(self, pmask, rmask, prob):
        loss_ce = self.ce(pmask, rmask)
        loss_ce = (loss_ce*prob).mean()
        loss_dice = self.dice(pmask, rmask)
        loss = loss_ce  + loss_dice
        return loss

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))