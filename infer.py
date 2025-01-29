import torch
from PIL import Image
from torchvision import transforms
from dataset import rgb_transform, depth_transform, depth2_transform
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.transforms.functional import resize
from Discriminative import MMM
from metrics import get_maxFscore_and_threshold
import pdb
import time
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = '/data1/kurohiji/dis/0129/trained_step3.pth'  # Path to the trained model
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
depth_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/test/depth_normalized'
depth2_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/test/depth2'
output_dir = 'output12/0129'
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

fbeta_score = 0
time_list = []
# Loop through all images
for rgb_image_name, depth_image_name, depth2_image_name, gt_image_name in zip(rgb_images, depth_images, depth2_images, gt_images):
    rgb_image_path = os.path.join(rgb_dir, rgb_image_name)
    depth_image_path = os.path.join(depth_dir, depth_image_name)
    depth2_image_path = os.path.join(depth2_dir, depth2_image_name)
    gt_image_path = os.path.join(gt_dir, gt_image_name)
    
    # Load images
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    depth_image = Image.open(depth_image_path).convert('L')
    depth2_image = Image.open(depth2_image_path).convert('L')
    gt_image = Image.open(gt_image_path).convert('L')
    
    # Apply transforms
    rgb_tensor = rgb_transform(rgb_image).unsqueeze(0).to(device)
    depth_tensor = depth_transform(depth_image).unsqueeze(0).to(device)
    depth2_tensor = depth2_transform(depth2_image).unsqueeze(0).to(device)
    
    # Perform inference
    start_each = time.time()
    with torch.no_grad():
        outputs_pm = model(rgb_tensor, depth_tensor, depth2_tensor,mode_flag=4)
    time_each = time.time() - start_each
    time_list.append(time_each)
    # Convert output to numpy array

    output_mask = outputs_pm.squeeze().cpu().numpy()
  
    gt_binary = np.array(gt_image.convert('1')).astype(int)
    output_mask_resized = resize(torch.tensor(output_mask).unsqueeze(0).unsqueeze(0), gt_binary.shape).squeeze().numpy()

    # Evaluate F-score
    pred_flat = output_mask_resized.flatten().astype(np.float32)
    true_flat = gt_binary.flatten().astype(np.uint8)
    Fbeta, thres = get_maxFscore_and_threshold(true_flat, pred_flat)
    fbeta_score += Fbeta

    # ----- 確率マップをカラーマップjetでカラーバー付きで保存 -----
    # Figureを生成
    fig, ax = plt.subplots()
    cax = ax.imshow(output_mask_resized, cmap='jet')
    fig.colorbar(cax)
    ax.axis('off')

    # ファイル名作成
    base_name = os.path.splitext(rgb_image_name)[0]
    prob_map_path = os.path.join(output_dir_prob, f'{base_name}_probability.png')
    plt.savefig(prob_map_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ----- 閾値処理後のバイナリマスクを保存 -----

    binary_mask = (output_mask_resized >= thres).astype(np.uint8) * 255
    binary_pil = Image.fromarray(binary_mask)
    bin_mask_path = os.path.join(output_dir_bin, f'{base_name}_binary.png')
    binary_pil.save(bin_mask_path)

    print(f'Probability map saved to {prob_map_path}')
    print(f'Binary mask saved to {bin_mask_path}')

print("{}'s average Time Is : {:.1f} ms".format(rgb_image_name, np.mean(time_list) * 1000))
print("{}'s average Time Is : {:.1f} fps".format(rgb_image_name, 1 / np.mean(time_list)))
Fbeta_mean = fbeta_score / len(rgb_images)
print("Mean F-beta:", Fbeta_mean)
