import torch
from PIL import Image
from torchvision import transforms
from dataset import rgb_transform, depth_transform, depth2_transform
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.transforms.functional import resize
from model import MMM
from metrics import compute_metrics
import imageio.v3 as iio
import pdb
import time
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = '/data1/kurohiji/dis/0211_2/trained_step3.pth'  # Path to the trained model
model = MMM().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
params = 0
for p in model.parameters():
    if p.requires_grad:
        params += p.numel()

print(params)  # 121898
# Directories for input images
rgb_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/test/image'
depth_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/test/depth'
depth2_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/test/depth2'
output_dir = 'output12/0211_2'
gt_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/test/mask_single'

# 追加の出力ディレクトリ
output_dir_prob = os.path.join(output_dir, 'probability_maps')
output_dir_bin = os.path.join(output_dir, 'binary_masks')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_prob, exist_ok=True)
os.makedirs(output_dir_bin, exist_ok=True)

# Get list of image files
rgb_images = sorted(os.listdir(rgb_dir))
depth_images = sorted(os.listdir(depth_dir))
depth2_images = sorted(os.listdir(depth2_dir))
gt_images = sorted(os.listdir(gt_dir))
# Ensure all directories have the same number of images
assert len(rgb_images) == len(depth_images) == len(depth2_images), "Mismatch in the number of images in directories"
fbeta_score_half = []
fbeta_score = []
time_list = []
# 追加: 評価結果の出力ディレクトリを作成
output_dir_eval = os.path.join(output_dir, 'evaluation_results')
os.makedirs(output_dir_eval, exist_ok=True)

# 必要なメトリクスの初期化
total_iou = []
total_mae = []
total_ber = []

cnt = 0

# ループ処理で各メトリクスを計算し、画像を保存
for  rgb_image_name, depth_image_name, depth2_image_name, gt_image_name in zip(rgb_images, depth_images, depth2_images, gt_images):
    #cnt += 1
    rgb_image_path = os.path.join(rgb_dir, rgb_image_name)
    depth_image_path = os.path.join(depth_dir, depth_image_name)
    depth2_image_path = os.path.join(depth2_dir, depth2_image_name)
    gt_image_path = os.path.join(gt_dir, gt_image_name)
    
    # 画像の読み込み
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    #depth_image = Image.open(depth_image_path).convert('L')
    depth2_image = Image.open(depth2_image_path).convert('L')
    gt_image = Image.open(gt_image_path).convert('L')

    depth = iio.imread(depth_image_path, mode='I')  # 16-bit uint16 のまま読み込む
    depth = depth.astype(np.float32) / 65535  # 正規化
    #pdb.set_trace()

    # PyTorch の警告を回避するためにコピーを作成
    depth_image = torch.from_numpy(depth.copy())  # (460, 620) → Tensor
    depth_image = depth_image.unsqueeze(0).numpy()  # (1, 460, 620)
    depth_image = (depth * 255).astype(np.uint8)  # 0-255 にスケール変換
    depth_image = Image.fromarray(depth_image)  # NumPy → PIL.Image
    
    # 変換の適用
    rgb_tensor = rgb_transform(rgb_image).unsqueeze(0).to(device)
    depth_tensor = depth_transform(depth_image).unsqueeze(0).to(device)
    depth2_tensor = depth2_transform(depth2_image).unsqueeze(0).to(device)
    
    # 推論の実行
    start_each = time.time()
    with torch.no_grad():
        outputs_pm = model(rgb_tensor, depth_tensor, depth2_tensor, mode_flag=4)
        #pdb.set_trace()
    time_each = time.time() - start_each
    time_list.append(time_each)
    
    # Resize output to match GT
    output_mask = outputs_pm.squeeze(0)[0]
    # GT画像のリサイズ
    gt_resized = resize(gt_image, output_mask.shape, interpolation=Image.BILINEAR)

    # バイナリマップに変換（閾値0.5を適用）
    gt_binary = (np.array(gt_resized).astype(np.float32) / 255.0) >= 0.5
    gt_tensor = torch.tensor(gt_binary.astype(np.float32), device=device)

    output_mask = torch.clamp(output_mask, 0, 1)
    #pdb.set_trace()

    output_mask_resized = resize(output_mask.unsqueeze(0).unsqueeze(0), np.array(gt_image).shape).squeeze(0).squeeze(0)
    output_mask_resized = torch.clamp(output_mask_resized, 0, 1)
    #pdb.set_trace()
    # Compute metrics
    metrics = compute_metrics(output_mask, gt_tensor)

    total_iou.append(metrics["IoU"])
    total_mae.append(metrics["MAE"])
    total_ber.append(metrics["BER"])
    fbeta_score_half.append(metrics["Max Fβ_0.5"])
    fbeta_score.append(metrics["Max Fβ_1.0"])
    #pdb.set_trace()

    output_mask_np = output_mask_resized.detach().cpu().numpy()

    # Optimal Threshold を float に変換
    optimal_threshold = float(metrics["Optimal Threshold"])

    # 閾値を適用してバイナリマスクを作成
    binary_mask = (output_mask_np >= optimal_threshold).astype(np.uint8)


    


    # ----- 確率マップをカラーマップ付きで保存 -----
    fig, ax = plt.subplots()
    cax = ax.imshow(output_mask_np, cmap='jet')
    fig.colorbar(cax)
    ax.axis('off')

    base_name = os.path.splitext(rgb_image_name)[0]
    prob_map_path = os.path.join(output_dir_prob, f'{base_name}_probability.png')
    plt.savefig(prob_map_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ----- バイナリマスクを保存 -----
    binary_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    bin_mask_path = os.path.join(output_dir_bin, f'{base_name}_binary.png')
    binary_pil.save(bin_mask_path)

    print(f'Probability map saved to {prob_map_path}')
    print(f'Binary mask saved to {bin_mask_path}')


pdb.set_trace()
# 平均値の計算
num_images = len(rgb_images)
Fbeta_half_mean = np.mean(fbeta_score_half)
Fbeta_mean = np.mean(fbeta_score)
IoU_mean = np.mean(total_iou)
MAE_mean = np.mean(total_mae)
BER_mean = np.mean(total_ber)
avg_time = np.mean(time_list) * 1000  # in milliseconds
fps = 1 / np.mean(time_list)

# 全体の平均評価結果をテキストファイルに保存
summary_result_path = os.path.join(output_dir_eval, 'summary_evaluation.txt')

with open(summary_result_path, 'w') as f:
    f.write("Overall Evaluation Results\n")
    f.write("==========================\n")
    f.write(f"Mean F-beta 0.5 Score: {Fbeta_half_mean:.4f}\n")
    f.write(f"Mean F-beta 1.0 Score: {Fbeta_mean:.4f}\n")
    f.write(f"Mean IoU: {IoU_mean:.4f}\n")
    f.write(f"Mean MAE: {MAE_mean:.4f}\n")
    f.write(f"Mean BER: {BER_mean:.4f}\n")
    f.write(f"Average Inference Time per Image: {avg_time:.2f} ms\n")
    f.write(f"Average FPS: {fps:.2f}\n")

print(f'Overall evaluation results saved to {summary_result_path}')
