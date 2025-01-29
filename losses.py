import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def iou_loss(preds, targets, smooth=1e-6):
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()

def sobel_filter():
    """Sobelフィルタを生成 (X方向とY方向)"""
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    kernel = torch.stack([kernel_x, kernel_y]).unsqueeze(1)  # (2, 1, 3, 3)
    return kernel


def generate_edge_map(mask):
    """
    GPU 上でエッジマップを生成 (Sobelフィルタを使用)
    """
    if len(mask.shape) == 5:  # 形状が [B, C, 1, H, W] の場合
        mask = mask.squeeze(2)

    device = mask.device
    kernel = sobel_filter().to(device)

    edge_x = F.conv2d(mask, kernel[0:1], padding=1)
    edge_y = F.conv2d(mask, kernel[1:2], padding=1)

    # エッジ強度の計算 (負値を防ぐ)
    edge = torch.sqrt(torch.clamp(edge_x**2 + edge_y**2, min=1e-8))

    # 正規化 (ゼロ割り防止)
    edge = edge / (edge.max() + 1e-8)

    return edge


def edge_loss(prediction, gt_edge, patch_size=8):

    # エッジマップを生成
    pred_edge = generate_edge_map(prediction)


    # パッチに分割 (unfoldを使用)
    pred_patches = pred_edge.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    gt_patches = gt_edge.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    # 値を範囲内に制限
    pred_patches = torch.clamp(pred_patches, 0, 1)
    gt_patches = torch.clamp(gt_patches, 0, 1)


    total_loss = torch.abs(pred_patches - gt_patches).mean()

    return total_loss

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
        self.focal_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou = iou_loss
        self.edge = edge_loss

    def forward(self, preds, targets,edge_targets):
        preds_sig = torch.sigmoid(preds)  # シグモイドを適用
        f_loss = self.focal_loss(preds, targets).mean()
        i_loss = self.iou(preds_sig, targets)
        e_loss = self.edge(preds_sig, edge_targets)
        return f_loss + i_loss + 10 * e_loss


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

