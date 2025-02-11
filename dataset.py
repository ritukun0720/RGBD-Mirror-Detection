# Updating the dataset.py to include depth2 images and adding resizing for depth2 images.
import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import imageio.v3 as iio

import pdb
# Ensure that PIL can handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, depth2_dir, mask_dir, rgb_transform=None, depth_transform=None, depth2_transform=None, mask_transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.depth2_dir = depth2_dir
        self.mask_dir = mask_dir

        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.depth_images = sorted(os.listdir(depth_dir))
        self.depth2_images = sorted(os.listdir(depth2_dir))
        self.mask_images = sorted(os.listdir(mask_dir))

        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.depth2_transform = depth2_transform
        self.mask_transform = mask_transform

        self.dilate = 1

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_images[idx])
        depth2_path = os.path.join(self.depth2_dir, self.depth2_images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])

        rgb_image = Image.open(rgb_path).convert('RGB')
        depth2_image = Image.open(depth2_path).convert('L')
        mask_image = Image.open(mask_path).convert('L')
        #pdb.set_trace()
        # 16-bit PNG を適切に読み込む
        depth = iio.imread(depth_path, mode='I')  # 16-bit uint16 のまま読み込む
        depth = depth.astype(np.float32) / 65535  # 正規化
        #pdb.set_trace()

        # PyTorch の警告を回避するためにコピーを作成
        depth_image = torch.from_numpy(depth.copy())  # (460, 620) → Tensor
        depth_image = depth_image.unsqueeze(0).numpy()  # (1, 460, 620)
        depth_image = (depth * 255).astype(np.uint8)  # 0-255 にスケール変換
        depth_image = Image.fromarray(depth_image)  # NumPy → PIL.Image
 

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)
        if self.depth2_transform:
            depth2_image = self.depth2_transform(depth2_image)
        if self.mask_transform:
            mask_image = self.mask_transform(mask_image)
        #depth_image = depth_image / 65535
        # エッジ画像の生成
        edge_canny = cv2.Canny(np.array(mask_image.squeeze(0), dtype=np.uint8) * 255, 0, 255)
        edge_canny[edge_canny < 127] = 0
        edge_canny[edge_canny > 127] = 1
        edge = cv2.dilate(edge_canny, np.ones((self.dilate, self.dilate)))
        edge = torch.tensor(np.expand_dims(edge, 0).astype(np.float32))

        filename = self.rgb_images[idx]  # 元画像のファイル名を取得

        return rgb_image, depth_image, depth2_image, mask_image, edge, filename  # **ファイル名を追加**


# Transformations for data augmentation and normalization
rgb_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

depth_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

depth2_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

