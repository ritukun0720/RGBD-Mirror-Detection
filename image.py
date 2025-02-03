import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import pdb
def plot_loss_gragh(t_loss, v_loss, t_score, v_score, save_path):
    # 保存先ディレクトリの作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(25,9))
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Loss", fontsize=18)
    ax1.set_xlabel("Epoch",fontsize=18)
    ax1.set_ylabel("Loss",fontsize=18)
    ax1.plot(t_loss, label="train", marker='o')
    ax1.plot(v_loss, label="valid", marker='o')
    ax1.tick_params(axis='both',labelsize=15)
    ax1.grid()
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("BinaryFbetaScore", fontsize=18)
    ax2.set_xlabel("Epoch", fontsize=18)
    ax2.set_ylabel("IoU", fontsize=18)
    ax2.plot(t_score, label="train", marker='o')
    ax2.plot(v_score, label="valid", marker='o')
    ax2.tick_params(axis='both',labelsize=15)
    ax2.grid()
    ax2.legend()
    plt.savefig(save_path)
    plt.close()

def save_epoch_images_grid(features, inputs_rgb, mask, save_path):
    """
    特徴マップと入力画像、マスクを2x4のレイアウトで保存。
    features: 特徴マップリスト [features1, features2, features3, features4]
    inputs_rgb: 入力RGB画像 (torch.Tensor)
    mask: マスク画像 (torch.Tensor)
    save_path: 保存先パス
    """
    # 特徴マップと画像を正規化して0-1にスケール変換
    def normalize_tensor(tensor):
        tensor = tensor.cpu().detach()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
        return tensor.numpy()

    # 正規化
    normalized_rgb = normalize_tensor(inputs_rgb.permute(1, 2, 0))  # RGB画像
    normalized_mask = normalize_tensor(mask.squeeze())  # マスク画像を2次元に変換
    normalized_features = [normalize_tensor(f.squeeze()) for f in features]  # 特徴マップを2次元に変換

    # レイアウトを設定
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2行×4列のグリッド
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # 各画像を配置
    # 1行目：特徴マップ
    for i, feature in enumerate(normalized_features):
        ax = axes[0, i]
        ax.imshow(feature, cmap="gray")
        ax.set_title(f"Feature {i+1}", fontsize=10)
        ax.axis("off")

    # 2行目：入力画像、マスク画像、その他空白
    axes[1, 0].imshow(normalized_rgb)
    axes[1, 0].set_title("Input Image", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(normalized_mask, cmap="gray")
    axes[1, 1].set_title("Ground Truth", fontsize=10)
    axes[1, 1].axis("off")

    # 空白で埋める
    for i in range(2, 4):
        axes[1, i].axis("off")

    # 残りの空白
    for i in range(4, 8):
        axes[0, i % 4].axis("off")

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_epoch_edge_grid(features, inputs_rgb, mask, save_path):
    """
    特徴マップと入力画像、マスクを2xNのレイアウトで保存。
    features: 特徴マップリスト [features1, features2, ...]
    inputs_rgb: 入力RGB画像 (torch.Tensor)
    mask: マスク画像 (torch.Tensor)
    save_path: 保存先パス
    """
    def normalize_tensor(tensor):
        tensor = tensor.cpu().detach()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
        return tensor

    # Normalize inputs
    normalized_rgb = normalize_tensor(inputs_rgb.permute(1, 2, 0)).numpy()
    normalized_mask = normalize_tensor(mask.squeeze()).numpy()

    # Use features without normalization
    raw_features = features.squeeze().cpu().detach().numpy()

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Feature map with jet colormap and colorbar
    im = axes[0].imshow(raw_features, cmap="jet")
    axes[0].set_title("Feature Map", fontsize=10)
    axes[0].axis("off")
    cbar = fig.colorbar(im, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Raw Value", fontsize=8)

    # Input RGB image
    axes[1].imshow(normalized_rgb)
    axes[1].set_title("Input Image", fontsize=10)
    axes[1].axis("off")

    # Ground truth mask
    axes[2].imshow(normalized_mask, cmap="gray")
    axes[2].set_title("Ground Truth", fontsize=10)
    axes[2].axis("off")

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()