import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import pdb

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

def bce2d_new(input, target, reduction='mean'):
        assert(input.size() == target.size())
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()
        # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg

        alpha = num_neg / num_total
        beta = 1.1 * num_pos / num_total
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = alpha * pos + beta * neg

        return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

def BCE_IOU(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return wbce.mean(), wiou.mean()

# --------------------------- BINARY Lovasz LOSSES ---------------------------
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


# def lovasz_hinge_flat(logits, labels):
#     """
#     Binary Lovasz hinge loss
#       logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
#       labels: [P] Tensor, binary ground truth labels (0 or 1)
#       ignore: label to ignore
#     """
#     if len(labels) == 0:
#         # only void pixels, the gradients should be 0
#         return logits.sum() * 0.
#     signs = 2. * labels.float() - 1.
#     errors = (1. - logits * Variable(signs))
#     errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
#     perm = perm.data
#     gt_sorted = labels[perm]
#     grad = lovasz_grad(gt_sorted)
#     loss = torch.dot(F.relu(errors_sorted), Variable(grad))
#     return loss

def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.
    Args:
        logits (torch.Tensor): [P], logits at each prediction
            (between -infty and +infty).
        labels (torch.Tensor): [P], binary ground truth labels (0 or 1).
    Returns:
        torch.Tensor: The calculated loss.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, labels.float())
    return loss

# def lovasz_grad(gt_sorted):
#     """
#     Computes gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     p = len(gt_sorted)
#     gts = gt_sorted.sum()
#     intersection = gts - gt_sorted.float().cumsum(0)
#     union = gts + (1 - gt_sorted).float().cumsum(0)
#     jaccard = 1. - intersection / union
#     if p > 1: # cover 1-pixel case
#         jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#     return jaccard

# new implementation from mmseg
# https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/losses/lovasz_loss.py

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def isnan(x):
    return x != x

class FocalMultiLabelLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=1000.0):
        """
        Args:
            gamma (float): フォーカスパラメータ。
            pos_weight (float): 正例の重み付け。
        """
        super(FocalMultiLabelLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight]))

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): モデルのロジット出力。形状は (batch_size, num_classes)。
            targets (Tensor): バイナリラベル (0 または 1)。形状は (batch_size, num_classes)。
        Returns:
            Tensor: 平均フォーカル損失。
        """
        # デバイスの設定（pos_weightをデバイスに移動）
        self.bce_with_logits.pos_weight = self.bce_with_logits.pos_weight.to(outputs.device)

        # BCE損失の計算
        bce_loss = self.bce_with_logits(outputs, targets)

        # 確率の計算 (ロジットからシグモイドを適用)
        probs = torch.sigmoid(outputs)

        # フォーカル損失の重み計算
        pt = torch.where(targets == 1, probs, 1 - probs)  # pt = P_t
        focal_weight = (1 - pt) ** self.gamma

        # フォーカル損失の計算
        focal_loss = focal_weight * bce_loss

        # 平均を取る
        return focal_loss.mean()

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou = lovasz_hinge


    def forward(self, preds, targets,edge_targets):
        
        b_loss = self.bce_loss(preds, targets).mean()
        i_loss = self.iou(preds, targets)
        return b_loss + i_loss


class featureloss(nn.Module):
    def __init__(self):
        super(featureloss, self).__init__()
        self.custom_loss = CustomLoss()



    def forward(self, features1,features2,features3,features4, targets,edge_targets):
        loss_dm1 = self.custom_loss(features1, targets,edge_targets)
        loss_dm2 = self.custom_loss(features2, targets,edge_targets)
        loss_dm3 = self.custom_loss(features3, targets,edge_targets)
        loss_dm4 = self.custom_loss(features4, targets,edge_targets)

        return loss_dm1 + 2 * loss_dm2 + 3 * loss_dm3 + 4 * loss_dm4

