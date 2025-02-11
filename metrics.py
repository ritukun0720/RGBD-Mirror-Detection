import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
from torchmetrics import JaccardIndex, MeanAbsoluteError

def compute_ber(preds_binary, targets):
    """
    Computes the Balanced Error Rate (BER).

    Args:
        preds_binary (torch.Tensor): Binarized predictions (0 or 1) with shape (batch, height, width).
        targets (torch.Tensor): Ground truth binary values (0 or 1) with shape (batch, height, width).

    Returns:
        float: BER score in percentage (0 to 100).
    """
    TP = torch.sum((preds_binary == 1) & (targets == 1)).float()
    TN = torch.sum((preds_binary == 0) & (targets == 0)).float()
    Np = torch.sum(targets == 1).float()  # 正例の総数
    Nn = torch.sum(targets == 0).float()  # 負例の総数

    TP_rate = TP / (Np + 1e-6)  # True Positive Rate
    TN_rate = TN / (Nn + 1e-6)  # True Negative Rate
    BER = (1 - 0.5 * (TP_rate + TN_rate)) * 100  # BER 計算

    return BER.item()

def get_maxFscore_and_threshold(true_1d, pred_1d, beta=0.5):
    """
    Computes the maximum Fβ score and the corresponding threshold.

    Args:
        true_1d (np.ndarray): Ground truth binary values (0 or 1).
        pred_1d (np.ndarray): Predicted probability values (0 to 1).
        beta (float): Beta value for the F-score.

    Returns:
        tuple: (max Fβ score, optimal threshold)
    """
    assert len(true_1d.shape) == 1 and len(pred_1d.shape) == 1, "Input shape must be 1-dimensional."
    
    precisions, recalls, thresholds = precision_recall_curve(true_1d, pred_1d)
    
    # Compute F-beta score
    numerator = (1 + beta ** 2) * (precisions * recalls)
    denom = (beta ** 2 * precisions) + recalls
    fbetas = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))

    return np.max(fbetas), thresholds[np.argmax(fbetas)]

def compute_metrics(preds, targets, threshold=0.5):
    """
    Computes IoU (Jaccard Index), MAE (Mean Absolute Error), BER (Balanced Error Rate), 
    and the max Fβ score with the optimal threshold.

    Args:
        preds (torch.Tensor): Predicted probability values (0 to 1) with shape (batch, height, width).
        targets (torch.Tensor): Ground truth binary values (0 or 1) with shape (batch, height, width).
        threshold (float): Threshold to binarize predictions for IoU and BER calculation.

    Returns:
        dict: {"IoU": float, "MAE": float, "BER": float, "Max Fβ": float, "Optimal Threshold": float}
    """
    preds_binary = (preds > threshold).float()
    targets = targets.float()

    # IoU 計算
    iou_metric = JaccardIndex(task="binary", num_classes=2).to(preds.device)
    iou_score = iou_metric(preds_binary, targets).item()

    # MAE 計算
    mae_metric = MeanAbsoluteError().to(preds.device)
    mae_score = mae_metric(preds, targets).item()

    # BER 計算（関数を呼び出し）
    ber_score = compute_ber(preds_binary, targets)

    # F-beta スコアと最適閾値の計算（テンソルのまま処理）
    max_fbeta_half, optimal_threshold = get_maxFscore_and_threshold(
        targets.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy()
    )
    max_fbeta,_= get_maxFscore_and_threshold(
        targets.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy(),beta=1.0
    )


    return {
        "IoU": iou_score,
        "MAE": mae_score,
        "BER": ber_score,
        "Max Fβ_0.5": max_fbeta_half,
        "Max Fβ_1.0": max_fbeta,
        "Optimal Threshold": optimal_threshold
    }
