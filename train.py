import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset, rgb_transform, depth_transform, depth2_transform, mask_transform
from losses import featureloss, FocalMultiLabelLoss, CustomLoss
from early_stopping import EarlyStopping
from Discriminative import MMM
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryFBetaScore
from image import plot_loss_gragh, save_epoch_images_grid, save_epoch_edge_grid
import pdb

# コマンドライン引数の解析
parser = argparse.ArgumentParser(description='Train Discriminative SubNetwork with adjustable output directory')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints and outputs')
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# データセットのロード
rgb_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/train/image'
depth_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/train/depth_normalized'
depth2_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/train/depth2'
mask_dir = '/data1/kurohiji/RGBD/CVPR2021_PDNet/data/RGBD-Mirror/train/mask_single'

dataset = CustomDataset(
    rgb_dir=rgb_dir,
    depth_dir=depth_dir,
    depth2_dir=depth2_dir,
    mask_dir=mask_dir,
    rgb_transform=rgb_transform,
    depth_transform=depth_transform,
    depth2_transform=depth2_transform,
    mask_transform=mask_transform
)

# データセットの分割
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=18, shuffle=False)

# デバイスの設定
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# モデル、損失関数、オプティマイザの初期化
model = MMM().to(device)
criterion1 = CustomLoss()
criterion2 = FocalMultiLabelLoss()
criterion3 = CustomLoss()
metrics_fn = BinaryFBetaScore(beta=0.5).to(device)
steps = [
    {
        "name": "mirror detection",
        "mode": {
            "refinement_net": False,
            "EdgeDetectionModel": False,
            "DiscriminativeSubNetwork": True
        },
        "criterion": criterion1,
        "optimizer": optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.001, momentum=0.9, weight_decay=5e-4
        ),
        "early_stopping_path": os.path.join(output_dir, 'trained_step1.pth'),
        "mode_flag": 1,
        "save_image_name": 'UNet.png',
        "previous_checkpoint": None
    },
    {
        "name": "Edge Detection",
        "mode": {
            "refinement_net": False,
            "EdgeDetectionModel": True,
            "DiscriminativeSubNetwork": False
        },
        "criterion": criterion2,
        "optimizer": optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.001, momentum=0.9, weight_decay=5e-4
        ),
        "early_stopping_path": os.path.join(output_dir, 'trained_step2.pth'),
        "mode_flag": 2,
        "save_image_name": 'edge.png',
        "previous_checkpoint": os.path.join(output_dir, 'trained_step1.pth')
    },
    {
        "name": "Refinement",
        "mode": {
            "refinement_net": True,
            "EdgeDetectionModel": False,
            "DiscriminativeSubNetwork": False
        },
        "criterion": criterion3,
        "optimizer": optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.001, momentum=0.9, weight_decay=5e-4
        ),
        "early_stopping_path": os.path.join(output_dir, 'trained_step3.pth'),
        "mode_flag": 3,
        "save_image_name": 'fin.png',
        "previous_checkpoint": os.path.join(output_dir, 'trained_step2.pth')
    }
]
start_ind = 0
for step_ind in range(start_ind,len(steps)):
    step = steps[step_ind]
    print(f"Starting phase: {step['name']}")
    
    # 前のステップのチェックポイントをロード
    if step['previous_checkpoint'] and os.path.exists(step['previous_checkpoint']):
        #pdb.set_trace()
        model.load_state_dict(torch.load(step['previous_checkpoint']))
        print(f"Loaded checkpoint from previous step: {step['previous_checkpoint']}")
    
    # 現在のステップの early stopping パスを設定
    early_stopping = EarlyStopping(
        patience=10, verbose=True, path=step['early_stopping_path']
    )

    for param_name, mode in step['mode'].items():
        for param in getattr(model, param_name).parameters():
            param.required_grad = mode

    optimizer = step['optimizer']

    train_loss, val_loss = [], []
    train_metrics, val_metrics = [], []

    for epoch in range(600):
        model.train()
        running_loss, train_score = 0.0, 0.0

        for i, data in enumerate(train_loader):
            inputs_rgb, inputs_depth, inputs_depth2, targets, edge_targets = data
            inputs_rgb, inputs_depth, inputs_depth2, targets = (
                inputs_rgb.to(device), inputs_depth.to(device),
                inputs_depth2.to(device), targets.to(device)
            )

            if step['name'] == "Edge Detection":
                edge_targets = edge_targets.to(device)

            optimizer.zero_grad()
            if step['name'] == "mirror detection":
                outputs4 = model(inputs_rgb, inputs_depth, inputs_depth2, mode_flag=step['mode_flag'])
                #outputs1, outputs2, outputs3, outputs4 = outputs
                loss = step['criterion'](outputs4, targets)
            elif step['name'] == "Edge Detection":
                outputs4 = model(inputs_rgb, inputs_depth, inputs_depth2, mode_flag=step['mode_flag'])
                loss = step['criterion'](outputs4, edge_targets)
            else:
                outputs4 = model(inputs_rgb, inputs_depth, inputs_depth2, mode_flag=step['mode_flag'])
                loss = step['criterion'](outputs4, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_score += metrics_fn(torch.sigmoid(outputs4), (targets if step['name'] != "Edge Detection" else edge_targets).int()).item()
            if i % 10 == 9:
                print(f'Epoch [{epoch + 1}/600], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / (i+1):.4f}')
            
        train_loss.append(running_loss / len(train_loader))
        train_metrics.append(train_score / len(train_loader))

        # Validation loop
        model.eval()
        valing_loss, val_score = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                inputs_rgb, inputs_depth, inputs_depth2, targets, edge_targets = data
                inputs_rgb, inputs_depth, inputs_depth2, targets = (
                    inputs_rgb.to(device), inputs_depth.to(device),
                    inputs_depth2.to(device), targets.to(device)
                )
                if step['name'] == "Edge Detection":
                    edge_targets = edge_targets.to(device)

                if step['name'] == "mirror detection":
                    outputs4 = model(inputs_rgb, inputs_depth, inputs_depth2, mode_flag=step['mode_flag'])
                    #outputs1, outputs2, outputs3, outputs4 = outputs
                    loss = step['criterion'](outputs4, targets)
                elif step['name'] == "Edge Detection":
                    outputs4 = model(inputs_rgb, inputs_depth, inputs_depth2, mode_flag=step['mode_flag'])
                    loss = step['criterion'](outputs4, edge_targets)
                else:
                    outputs4 = model(inputs_rgb, inputs_depth, inputs_depth2, mode_flag=step['mode_flag'])
                    loss = step['criterion'](outputs4, targets)


                valing_loss += loss.item()
                val_score += metrics_fn(torch.sigmoid(outputs4), (targets if step['name'] != "Edge Detection" else edge_targets).int()).item()

                if batch_idx == 0:
                    save_path = os.path.join(output_dir, 'val_images', step['save_image_name'])
                    if step['name'] == "Edge Detection":
                        save_epoch_edge_grid(torch.sigmoid(outputs4), inputs_rgb, edge_targets, save_path)
                    #elif step['name'] == "mirror detection":
                        #save_epoch_images_grid([outputs1, outputs2, outputs3, outputs4], inputs_rgb, targets, save_path)
                    else:     
                        save_epoch_edge_grid(torch.sigmoid(outputs4), inputs_rgb, targets, save_path)                        
                       
        val_loss.append(valing_loss / len(val_loader))
        val_metrics.append(val_score / len(val_loader))

        plot_loss_gragh(
            train_loss, val_loss, train_metrics, val_metrics, 
            os.path.join(output_dir, f'loss_g/loss_{step["name"]}.png')
        )

        print(f'Epoch [{epoch + 1}/600], Validation Loss: {valing_loss / len(val_loader):.4f}')
        early_stopping(valing_loss / len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"Phase {step['name']} completed!")
