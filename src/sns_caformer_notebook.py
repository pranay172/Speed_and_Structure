
# %% [markdown]
# # Speed and Structure: CAFormer Full Resolution Model
# This notebook adapts the high-performing CAFormer architecture from a similar Kaggle competition to the specific requirements of the Speed and Structure (SNS) challenge.
# 
# ### Core Adaptations:
# 1.  **Custom Stem:** A `HybridStem` is used to transform the SNS input `(10001, 31)` into a near-square feature map `(125, 124)` suitable for the `timm` backbone.
# 2.  **Custom Decoder:** A U-Net style decoder is re-architected to upsample features to the final SNS target shape of `(300, 1259)`, respecting the physical mapping of time-to-depth and receivers-to-horizontal.
# 3.  **Advanced Components:** It incorporates the best architectural features from the reference notebooks, including `PixelShuffle` upsampling, `SCSE` attention modules, and intermediate convolutions in the decoder.
# 4.  **Training Techniques:** It includes critical training enhancements like a `ConstantCosineLR` scheduler, Exponential Moving Average (EMA) of model weights, and Test-Time Augmentation (TTA) for robust performance.

# %% [markdown]
# ### 1. Setup and Imports

# %%
# !pip install --no-deps timm monai -q

import os

# Repository root (parent of src/); override with the SNS_ROOT environment variable.
PROJECT_ROOT = os.environ.get("SNS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import math
import random
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

import timm
from monai.networks.blocks import UpSample, SubpixelUpsample

# %% [markdown]
# ### 2. Configuration

# %%
cfg = SimpleNamespace()

# --- General ---
cfg.seed = 42
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
# IMPORTANT: Update these paths to your actual data directories
cfg.train_data_dir = os.path.join(PROJECT_ROOT, "data/train")
cfg.test_data_dir = os.path.join(PROJECT_ROOT, "data/test")

# --- Model ---
cfg.backbone = "caformer_b36.sail_in22k_ft_in1k"
cfg.pretrained = True
cfg.in_channels_stem = 5
cfg.stem_out_channels = 128 # Must match backbone's in_chans

# --- Training ---
cfg.epochs = 50
cfg.batch_size = 8 # Adjust based on your GPU memory
cfg.lr = 1e-4
cfg.weight_decay = 1e-5
cfg.ema_decay = 0.99

# --- Physics-Informed Loss ---
cfg.tv_loss_weight = 1e-4

# --- Scheduler ---
cfg.pct_cosine = 0.7 # Percentage of steps to use cosine annealing

# --- Logging & Saving ---
cfg.logging_steps = 50
cfg.save_path = os.path.join(PROJECT_ROOT, "models/caformer_sns_best_ema.pth")
cfg.latest_checkpoint_path = os.path.join(PROJECT_ROOT, "models/caformer_sns_latest.pth")
cfg.resume_checkpoint = None # Set to cfg.latest_checkpoint_path to resume

# %% [markdown]
# ### 3. Dataset Class

# %%
class SnsDataset(Dataset):
    """
    Custom dataset for the Speed and Structure (SNS) competition.
    """
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.sample_ids = [os.path.basename(p) for p in glob.glob(os.path.join(data_dir, '*'))]
        self.is_train = is_train
        self.source_coordinates = [1, 75, 150, 225, 300]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.data_dir, sample_id)

        seismic_inputs = []
        for i in self.source_coordinates:
            file_path = os.path.join(sample_path, f'receiver_data_src_{i}.npy')
            seismic_data = np.load(file_path).astype(np.float32)
            seismic_inputs.append(seismic_data)
        
        seismic_tensor = torch.from_numpy(np.stack(seismic_inputs, axis=0))

        if self.is_train:
            target_path = os.path.join(sample_path, 'vp_model.npy')
            target_tensor = torch.from_numpy(np.load(target_path).astype(np.float32))
            target_tensor = target_tensor.unsqueeze(0)

            # Augmentations
            if torch.rand(1) < 0.5:
                seismic_tensor = torch.flip(seismic_tensor, dims=[2]) # Horizontal flip of receivers
                target_tensor = torch.flip(target_tensor, dims=[1]) # Horizontal flip of velocity map
        else:
            # For the test set, we don't have targets. Return a placeholder.
            target_tensor = torch.empty(0)

        return seismic_tensor, target_tensor

# %% [markdown]
# ### 4. Model Architecture

# %%
class HybridStem(nn.Module):
    def __init__(self, in_channels=5, mid_channels=64, out_channels=128):
        super().__init__()
        self.downsample_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(10, 3), stride=(10, 1), padding=(0, 1)),
            nn.InstanceNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(4, 1), padding=(0, 1)),
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.Conv2d(64, mid_channels, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)),
            nn.InstanceNorm2d(mid_channels), nn.GELU()
        )
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(mid_channels), nn.GELU(),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(out_channels), nn.GELU()
        )

    def forward(self, x):
        x = self.downsample_block(x)
        x = self.upsample_block(x)
        return x

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, scale_factor=(2,2), upsample_mode="pixelshuffle"):
        super().__init__()

        conv_in_ch = in_ch // (int(scale_factor[0])**2) if upsample_mode == "pixelshuffle" else in_ch

        if upsample_mode == "pixelshuffle":
            self.upsample = SubpixelUpsample(spatial_dims=2, in_channels=in_ch, out_channels=conv_in_ch, scale_factor=int(scale_factor[0]))
        else:
            self.upsample = UpSample(spatial_dims=2, in_channels=in_ch, scale_factor=scale_factor, mode=upsample_mode)
        
        self.intermediate_conv = nn.Sequential(
            ConvBnAct(skip_ch, skip_ch, 3, 1),
            ConvBnAct(skip_ch, skip_ch, 3, 1),
        ) if skip_ch > 0 else nn.Identity()
        
        self.conv1 = ConvBnAct(conv_in_ch + skip_ch, out_ch, 3, 1)
        self.conv2 = ConvBnAct(out_ch, out_ch, 3, 1)
        self.attention = SCSEModule(out_ch)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            skip = self.intermediate_conv(skip)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x

class SnsDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0] # Add 0 for the last block which has no skip
        
        self.blocks = nn.ModuleList()
        for i_ch, s_ch, o_ch in zip(in_channels, skip_channels, decoder_channels):
            self.blocks.append(DecoderBlock(i_ch, s_ch, o_ch))
            
        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        skips = features[1:] + [None] # Add None for the last block
        
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
            
        x = self.final_conv(x)
        x = F.interpolate(x, size=(1259, 300), mode='bilinear', align_corners=False)
        x = torch.transpose(x, 2, 3)
        return x

class SnsCaformerModel(nn.Module):
    def __init__(self, backbone_name, pretrained=True, in_channels_stem=5, out_channels_stem=128):
        super().__init__()
        self.stem = HybridStem(in_channels=in_channels_stem, out_channels=out_channels_stem)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=out_channels_stem)
        self.backbone.stem = nn.Identity()

        # --- Dynamically get encoder channels to prevent size mismatch errors ---
        with torch.no_grad():
            dummy_input = torch.randn(2, in_channels_stem, 10001, 31)
            features = self.backbone(self.stem(dummy_input))
            encoder_channels = [f.shape[1] for f in features]
        # --- End dynamic channel getting ---

        self.decoder = SnsDecoder(encoder_channels=encoder_channels)

    def proc_flip(self, x):
        x_flipped = torch.flip(x, dims=[3])
        features = self.backbone(self.stem(x_flipped))
        prediction_flipped = self.decoder(features)
        prediction_unflipped = torch.flip(prediction_flipped, dims=[2])
        return prediction_unflipped

    def forward(self, x):
        features = self.backbone(self.stem(x))
        prediction = self.decoder(features)
        if not self.training:
            prediction_tta = self.proc_flip(x)
            prediction = (prediction + prediction_tta) / 2
        return prediction

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v.to(ema_v.device)))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

# %% [markdown]
# ### 5. Loss, Scheduler, and Training Loop

# %%
def mape_loss(prediction, target):
    epsilon = 1e-8
    return torch.mean(torch.abs((target - prediction) / (target.abs() + epsilon)))

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        # Assumes img is (N, C, H, W)
        h_tv = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return h_tv + w_tv

class CompositeLoss(nn.Module):
    def __init__(self, tv_weight=1e-4):
        super(CompositeLoss, self).__init__()
        self.tv_weight = tv_weight
        self.mape_loss = mape_loss
        self.tv_loss = TotalVariationLoss()

    def forward(self, prediction, target):
        mape = self.mape_loss(prediction, target)
        tv = self.tv_loss(prediction)
        combined_loss = mape + self.tv_weight * tv
        return combined_loss, {"mape": mape, "tv": tv}

class ConstantCosineLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, pct_cosine, last_epoch=-1):
        self.total_steps = total_steps
        self.milestone = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(total_steps - self.milestone, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        return [lr * factor for lr in self.base_lrs]

def run_training():
    # --- Setup ---
    torch.manual_seed(cfg.seed)
    if not os.path.exists(os.path.dirname(cfg.save_path)):
        os.makedirs(os.path.dirname(cfg.save_path))

    # --- Data ---
    full_dataset = SnsDataset(data_dir=cfg.train_data_dir, is_train=True)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, Optimizer, Loss, Scheduler ---
    model = SnsCaformerModel(backbone_name=cfg.backbone, pretrained=cfg.pretrained).to(cfg.device)
    ema_model = ModelEMA(model, decay=cfg.ema_decay)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = ConstantCosineLR(optimizer, total_steps=total_steps, pct_cosine=cfg.pct_cosine)
    criterion = CompositeLoss(tv_weight=cfg.tv_loss_weight)
    scaler = torch.amp.GradScaler('cuda' if cfg.device.type == 'cuda' else 'cpu')

    # --- Resume from Checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    if cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"Resuming training from {cfg.resume_checkpoint}")
        checkpoint = torch.load(cfg.resume_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.module.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")

    # --- Training Loop ---
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if cfg.device.type == 'cuda' else 'cpu'):
                predictions = model(inputs)
                loss, loss_components = criterion(predictions, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            ema_model.update(model)
            pbar.set_postfix({'Total Loss': f'{loss.item():.4f}', 'MAPE': f'{loss_components["mape"].item():.4f}', 'TV': f'{loss_components["tv"].item():.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        # --- Validation Loop ---
        ema_model.module.eval()
        val_loss, val_mape, val_tv = 0.0, 0.0, 0.0
        vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in vbar:
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                with torch.amp.autocast('cuda' if cfg.device.type == 'cuda' else 'cpu'):
                    predictions = ema_model.module(inputs)
                    loss, loss_components = criterion(predictions, targets)
                val_loss += loss.item()
                val_mape += loss_components["mape"].item()
                val_tv += loss_components["tv"].item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = val_mape / len(val_loader)
        avg_val_tv = val_tv / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Avg Val Total Loss: {avg_val_loss:.4f}, Avg Val MAPE: {avg_val_mape:.4f}, Avg Val TV: {avg_val_tv:.4f}")

        # --- Save Checkpoints ---
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }, cfg.latest_checkpoint_path)

        # Save best model
        if avg_val_mape < best_val_loss: # We care most about the MAPE for the final score
            best_val_loss = avg_val_mape
            torch.save(ema_model.module.state_dict(), cfg.save_path)
            print(f"New best EMA model saved to {cfg.save_path} with MAPE: {best_val_loss:.4f}")

# %% [markdown]
# ### 6. Run Training


# %% [markdown]
# ### 7. Inference and Submission

# %%
def create_submission(sample_id: str, prediction: np.ndarray, submission_path: str):
    try:
        submission = dict(np.load(submission_path))
    except FileNotFoundError:
        submission = dict({})
    submission.update({sample_id: prediction})
    np.savez(submission_path, **submission)

def run_inference():
    print("Starting inference...")
    model = SnsCaformerModel(backbone_name=cfg.backbone, pretrained=False).to(cfg.device)
    model.load_state_dict(torch.load(cfg.save_path))
    model.eval()

    test_dataset = SnsDataset(data_dir=cfg.test_data_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    submission_file = "submission.npz"
    if os.path.exists(submission_file):
        os.remove(submission_file)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(test_loader, desc="Inference")):
            inputs = inputs.to(cfg.device)
            predictions = model(inputs)
            
            # Get sample IDs for this batch
            start_idx = i * cfg.batch_size
            end_idx = start_idx + len(inputs)
            batch_sample_ids = test_dataset.sample_ids[start_idx:end_idx]

            for sample_id, pred_tensor in zip(batch_sample_ids, predictions):
                # Ensure prediction is float64 for submission
                prediction_np = pred_tensor.squeeze(0).cpu().numpy().astype(np.float64)
                create_submission(sample_id, prediction_np, submission_file)

    print(f"Inference complete. Submission file created at: {submission_file}")

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="SNS CAFormer Training")
    argument_parser.add_argument('--train', action='store_true', help='Run training')
    argument_parser.add_argument('--inference', action='store_true', help='Run inference')
    args = argument_parser.parse_args()
    if args.train:
        run_training()
    elif args.inference:
        run_inference()