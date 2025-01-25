import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def iou_loss(preds, targets, smooth=1e-6):
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()

def edge_loss(preds, targets):
    kernel = torch.tensor([[[[1, 1, 1],
                             [1, -9, 1],
                             [1, 1, 1]]]], dtype=torch.float32).to(preds.device)
    preds_edge = F.conv2d(preds, kernel, padding=1)
    targets_edge = F.conv2d(targets, kernel, padding=1)
    return F.l1_loss(preds_edge, targets_edge)

class FocalMultiLabelLoss(nn.Module):
    def __init__(self, gamma=5.0, pos_weight=5000.0):
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
        self.focal_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou = iou_loss
        self.edge = edge_loss

    def forward(self, preds, targets):
        preds_sig = torch.sigmoid(preds)  # シグモイドを適用
        f_loss = self.focal_loss(preds, targets).mean()
        i_loss = self.iou(preds_sig, targets)
        e_loss = self.edge(preds_sig, targets)
        return f_loss + i_loss + 10 * e_loss


class featureloss(nn.Module):
    def __init__(self):
        super(featureloss, self).__init__()
        self.custom_loss = CustomLoss()



    def forward(self, features1,features2,features3,features4, targets):
        loss_dm1 = self.custom_loss(features1, targets)
        loss_dm2 = self.custom_loss(features2, targets)
        loss_dm3 = self.custom_loss(features3, targets)
        loss_dm4 = self.custom_loss(features4, targets)

        return loss_dm1 + 2 * loss_dm2 + 3 * loss_dm3 + 4 * loss_dm4

