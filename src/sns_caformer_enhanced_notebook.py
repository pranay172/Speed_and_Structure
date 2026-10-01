
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
import logging
import gc
import warnings
from typing import Optional, Tuple, List, Dict, Union, Callable, Iterator, Generator
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
cfg.seed = 123
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
cfg.tv_loss_weight = 0.01  # Total Variation - geological smoothness
cfg.reconstruction_loss_weight = 0.1  # Multi-task reconstruction
cfg.smoothness_loss_weight = 0.005  # Local smoothness constraint
cfg.gradient_loss_weight = 0.002  # Realistic velocity gradients

# --- Adaptive Loss Scheduling ---
cfg.use_adaptive_physics_weights = True  # Enable adaptive weighting
cfg.physics_warmup_epochs = 10  # Epochs to ramp up physics losses
cfg.physics_max_weight_multiplier = 2.0  # Maximum weight increase

# --- Enhanced features ---
cfg.use_multi_task = True # Use multi-task learning with seismic reconstruction
cfg.use_tta = True # Use Test-Time Augmentation during inference

# --- Hyperparameters (moved from hardcoded values) ---
cfg.velocity_clip_min = 1.0 # Minimum velocity value (km/s)
cfg.velocity_clip_max = 8.0 # Maximum velocity value (km/s)
cfg.gradient_threshold = 3.0 # Maximum allowed gradient for quality validation

# --- Scheduler ---
cfg.pct_cosine = 0.7 # Percentage of steps to use cosine annealing

# --- Logging & Saving ---
cfg.logging_steps = 50
cfg.log_level = "INFO" # Logging level
cfg.save_path = os.path.join(PROJECT_ROOT, "models/caformer_sns_best_ema.pth")
cfg.latest_checkpoint_path = os.path.join(PROJECT_ROOT, "models/caformer_sns_latest.pth")
cfg.resume_checkpoint = None # Set to cfg.latest_checkpoint_path to resume

def validate_config(config: SimpleNamespace) -> None:
    """Validate configuration parameters and raise errors for invalid values."""
    # Basic parameter validation
    assert config.epochs > 0, f"epochs must be positive, got {config.epochs}"
    assert config.batch_size > 0, f"batch_size must be positive, got {config.batch_size}"
    assert config.lr > 0, f"learning rate must be positive, got {config.lr}"
    assert config.weight_decay >= 0, f"weight_decay must be non-negative, got {config.weight_decay}"
    assert 0 < config.ema_decay < 1, f"ema_decay must be in (0,1), got {config.ema_decay}"
    
    # Loss weight validation
    assert config.tv_loss_weight >= 0, f"tv_loss_weight must be non-negative, got {config.tv_loss_weight}"
    assert config.reconstruction_loss_weight >= 0, f"reconstruction_loss_weight must be non-negative, got {config.reconstruction_loss_weight}"
    assert config.smoothness_loss_weight >= 0, f"smoothness_loss_weight must be non-negative, got {config.smoothness_loss_weight}"
    assert config.gradient_loss_weight >= 0, f"gradient_loss_weight must be non-negative, got {config.gradient_loss_weight}"
    
    # Adaptive physics loss validation
    if hasattr(config, 'physics_warmup_epochs'):
        assert config.physics_warmup_epochs >= 0, f"physics_warmup_epochs must be non-negative, got {config.physics_warmup_epochs}"
        assert config.physics_warmup_epochs < config.epochs, f"physics_warmup_epochs ({config.physics_warmup_epochs}) must be less than total epochs ({config.epochs})"
    
    if hasattr(config, 'physics_max_weight_multiplier'):
        assert config.physics_max_weight_multiplier >= 1.0, f"physics_max_weight_multiplier must be >= 1.0, got {config.physics_max_weight_multiplier}"
    
    # Scheduler validation
    assert 0 < config.pct_cosine <= 1, f"pct_cosine must be in (0,1], got {config.pct_cosine}"
    
    # Hyperparameter validation
    assert config.gradient_threshold > 0, f"gradient_threshold must be positive, got {config.gradient_threshold}"
    
    # Directory validation
    if hasattr(config, 'train_data_dir') and config.train_data_dir:
        if not os.path.exists(config.train_data_dir):
            warnings.warn(f"Training data directory does not exist: {config.train_data_dir}")
    
    if hasattr(config, 'test_data_dir') and config.test_data_dir:
        if not os.path.exists(config.test_data_dir):
            warnings.warn(f"Test data directory does not exist: {config.test_data_dir}")
    
    # Model architecture validation
    assert config.in_channels_stem > 0, f"in_channels_stem must be positive, got {config.in_channels_stem}"
    assert config.stem_out_channels > 0, f"stem_out_channels must be positive, got {config.stem_out_channels}"
    

def setup_logging(config: SimpleNamespace) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {config.log_level} level")
    return logger

# Validate configuration
validate_config(cfg)
logger = setup_logging(cfg)

# %% [markdown]
# ### 3. Dataset Class

# %%
class SnsDataset(Dataset):
    """
    Custom dataset for the Speed and Structure (SNS) competition.
    
    Args:
        data_dir: Path to the data directory containing sample folders
        is_train: Whether this is training data (has targets) or test data
    """
    def __init__(self, data_dir: str, is_train: bool = True) -> None:
        self.data_dir = data_dir
        self.sample_ids = [os.path.basename(p) for p in glob.glob(os.path.join(data_dir, '*'))]
        self.is_train = is_train
        self.source_coordinates = [1, 75, 150, 225, 300]
        
        if len(self.sample_ids) == 0:
            raise ValueError(f"No samples found in directory: {data_dir}")
        
        logger.info(f"Initialized {'training' if is_train else 'test'} dataset with {len(self.sample_ids)} samples")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.data_dir, sample_id)

        seismic_inputs = []
        expected_seismic_shape = (10001, 31)  # Expected shape for receiver data
        
        for i in self.source_coordinates:
            file_path = os.path.join(sample_path, f'receiver_data_src_{i}.npy')
            try:
                seismic_data = np.load(file_path).astype(np.float32)
                
                # Validate shape
                if seismic_data.shape != expected_seismic_shape:
                    raise ValueError(f"Invalid seismic data shape in {file_path}: "
                                   f"expected {expected_seismic_shape}, got {seismic_data.shape}")
                
                # Validate data quality
                if not np.isfinite(seismic_data).all():
                    raise ValueError(f"Non-finite values detected in {file_path}")
                
                if seismic_data.dtype != np.float32:
                    logger.warning(f"Unexpected dtype in {file_path}: {seismic_data.dtype}, converting to float32")
                
                seismic_inputs.append(seismic_data)
                
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing seismic data file: {file_path}")
            except Exception as e:
                raise RuntimeError(f"Error loading seismic data from {file_path}: {e}")
        
        seismic_tensor = torch.from_numpy(np.stack(seismic_inputs, axis=0))

        if self.is_train:
            target_path = os.path.join(sample_path, 'vp_model.npy')
            expected_target_shape = (300, 1259)  # Expected shape for velocity model
            
            try:
                target_data = np.load(target_path).astype(np.float32)
                
                # Validate target shape
                if target_data.shape != expected_target_shape:
                    raise ValueError(f"Invalid target shape in {target_path}: "
                                   f"expected {expected_target_shape}, got {target_data.shape}")
                
                # Validate target data quality
                if not np.isfinite(target_data).all():
                    raise ValueError(f"Non-finite values detected in target {target_path}")
                
                # Validate velocity range (should be positive and reasonable)
                if target_data.min() <= 0:
                    raise ValueError(f"Non-positive velocities detected in {target_path}: min={target_data.min()}")
                
                if target_data.max() > 15.0:  # Very high but not impossible for seismic velocities
                    logger.warning(f"Very high velocities detected in {target_path}: max={target_data.max()}")
                
                target_tensor = torch.from_numpy(target_data)
                target_tensor = target_tensor.unsqueeze(0)
                
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing target file: {target_path}")
            except Exception as e:
                raise RuntimeError(f"Error loading target data from {target_path}: {e}")

            # Corrected horizontal flipping augmentation
            if torch.rand(1) < 0.5:
                # Flip receivers (spatial dimension)
                seismic_tensor = torch.flip(seismic_tensor, dims=[2])
                # Flip source order (channels) to match spatial flip
                seismic_tensor = torch.flip(seismic_tensor, dims=[0])
                # Flip velocity map horizontally
                target_tensor = torch.flip(target_tensor, dims=[1])
        else:
            # For the test set, we don't have targets. Return a placeholder.
            target_tensor = torch.empty(0)

        return seismic_tensor, target_tensor

# %% [markdown]
# ### 4. Model Architecture

# %%
class HybridStem(nn.Module):
    """Hybrid stem for processing seismic input data."""
    
    def __init__(self, in_channels: int = 5, mid_channels: int = 64, out_channels: int = 128) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class MultiTaskSnsDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32), use_multi_task=True):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        self.use_multi_task = use_multi_task
        
        # Main velocity prediction decoder
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        
        self.blocks = nn.ModuleList()
        for i_ch, s_ch, o_ch in zip(in_channels, skip_channels, decoder_channels):
            self.blocks.append(DecoderBlock(i_ch, s_ch, o_ch))
        
        # Velocity prediction head
        self.velocity_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        
        # Auxiliary seismic reconstruction head (only if multi-task is enabled)
        if self.use_multi_task:
            self.recon_head = nn.Sequential(
                nn.Conv2d(decoder_channels[-1], 64, kernel_size=3, padding=1),
                nn.InstanceNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 5, kernel_size=1)  # 5 channels for 5 sources
            )
        else:
            self.recon_head = None
    
    def forward(self, features):
        features = features[::-1]
        x = features[0]
        skips = features[1:] + [None]
        
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
        
        # Velocity prediction at full resolution
        velocity_pred = self.velocity_head(x)
        velocity_pred = F.interpolate(velocity_pred, size=(1259, 300), mode='bilinear', align_corners=False)
        velocity_pred = torch.transpose(velocity_pred, 2, 3)
        
        # Seismic reconstruction (for auxiliary task, only if multi-task is enabled)
        if self.use_multi_task and self.recon_head is not None:
            seismic_recon = self.recon_head(x)
            seismic_recon = F.interpolate(seismic_recon, size=(10001, 31), mode='bilinear', align_corners=False)
        else:
            seismic_recon = None
        
        return velocity_pred, seismic_recon

class MultiTaskSnsCaformerModel(nn.Module):
    def __init__(self, backbone_name, pretrained=True, in_channels_stem=5, out_channels_stem=128, 
                 use_multi_task=True, use_tta=True):
        super().__init__()
        self.use_multi_task = use_multi_task
        self.default_use_tta = use_tta
        self.stem = HybridStem(in_channels=in_channels_stem, out_channels=out_channels_stem)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=out_channels_stem)
        self.backbone.stem = nn.Identity()
        
        # Get encoder channels dynamically
        with torch.no_grad():
            dummy_input = torch.randn(2, in_channels_stem, 10001, 31)
            features = self.backbone(self.stem(dummy_input))
            encoder_channels = [f.shape[1] for f in features]
        
        self.decoder = MultiTaskSnsDecoder(encoder_channels=encoder_channels, use_multi_task=use_multi_task)
    
    def forward(self, x: torch.Tensor, return_auxiliary: bool = False, use_tta: bool = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        features = self.backbone(self.stem(x))
        velocity_pred, seismic_recon = self.decoder(features)
        
        # Use TTA if specified or use stored default
        if use_tta is None:
            use_tta = getattr(self, 'default_use_tta', True)
        
        if not self.training and not return_auxiliary and use_tta:
            # TTA during inference
            assert x.shape[2:] == (10001, 31), f'Expected input shape (*, *, 10001, 31), got: {x.shape}'
            velocity_pred_tta, _ = self.proc_flip(x)
            velocity_pred = (velocity_pred + velocity_pred_tta) / 2
            assert velocity_pred.shape[2:] == (300, 1259), f'Expected output shape (*, *, 300, 1259), got: {velocity_pred.shape}'
            logger.debug("Applied TTA during inference")
        
        if return_auxiliary:
            return velocity_pred, seismic_recon
        return velocity_pred
    
    def proc_flip(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply horizontal flip for test-time augmentation with corrected logic."""
        # Flip receivers (spatial dimension)
        x_flipped = torch.flip(x, dims=[3])
        # Flip source order (channels) to match spatial flip
        x_flipped = torch.flip(x_flipped, dims=[1])
        
        features = self.backbone(self.stem(x_flipped))
        velocity_pred_flipped, seismic_recon_flipped = self.decoder(features)
        
        # Flip velocity prediction back to original orientation
        velocity_pred_unflipped = torch.flip(velocity_pred_flipped, dims=[2])
        return velocity_pred_unflipped, seismic_recon_flipped

class MultiTaskLoss(nn.Module):
    def __init__(self, velocity_weight=1.0, reconstruction_weight=0.1, use_multi_task=True, **physics_weights):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.reconstruction_weight = reconstruction_weight
        self.use_multi_task = use_multi_task
        self.velocity_loss = EnhancedCompositeLoss(**physics_weights)
        if self.use_multi_task:
            self.reconstruction_loss = nn.L1Loss()
        
        logger.info(f"Initialized MultiTaskLoss with physics weights: {physics_weights}")
    
    def forward(self, velocity_pred, seismic_recon, velocity_target, seismic_input):
        # Main velocity prediction loss
        vel_loss, vel_components = self.velocity_loss(velocity_pred, velocity_target)
        
        # Auxiliary seismic reconstruction loss (only if multi-task is enabled)
        if self.use_multi_task and seismic_recon is not None:
            recon_loss = self.reconstruction_loss(seismic_recon, seismic_input)
            total_loss = (self.velocity_weight * vel_loss + 
                         self.reconstruction_weight * recon_loss)
            return total_loss, {
                **vel_components,
                "reconstruction": recon_loss,
                "total": total_loss
            }
        else:
            return vel_loss, {
                **vel_components,
                "reconstruction": torch.tensor(0.0, device=velocity_pred.device),
                "total": vel_loss
            }

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

class EnhancedCompositeLoss(nn.Module):
    def __init__(self, tv_weight=0.01, smoothness_weight=0.005, gradient_weight=0.002, 
                 use_adaptive_weights=True, current_epoch=0, 
                 physics_warmup_epochs=10, physics_max_weight_multiplier=2.0, epochs=50):
        super(EnhancedCompositeLoss, self).__init__()
        self.tv_weight = tv_weight
        self.smoothness_weight = smoothness_weight
        self.gradient_weight = gradient_weight
        self.use_adaptive_weights = use_adaptive_weights
        self.current_epoch = current_epoch
        
        # Store parameters instead of accessing cfg
        self.physics_warmup_epochs = physics_warmup_epochs
        self.physics_max_weight_multiplier = physics_max_weight_multiplier
        self.epochs = epochs
        
        self.mape_loss = mape_loss
        self.tv_loss = TotalVariationLoss()
    
    def update_epoch(self, epoch: int) -> None:
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch
    
    def get_adaptive_weight(self, base_weight: float, epoch: int) -> float:
        """Calculate adaptive weight based on training progress."""
        if not self.use_adaptive_weights:
            return base_weight
        
        # Warm up physics losses gradually
        if epoch < self.physics_warmup_epochs:
            # Avoid division by zero when physics_warmup_epochs is 0
            if self.physics_warmup_epochs == 0:
                warmup_factor = 1.0
            else:
                warmup_factor = epoch / self.physics_warmup_epochs
        else:
            # Gradually increase physics weight importance
            # Handle case where epochs equals physics_warmup_epochs (avoid division by zero)
            if self.epochs == self.physics_warmup_epochs:
                progress = 1.0
            else:
                progress = min((epoch - self.physics_warmup_epochs) / (self.epochs - self.physics_warmup_epochs), 1.0)
            warmup_factor = 1.0 + progress * (self.physics_max_weight_multiplier - 1.0)
        
        return base_weight * warmup_factor
    
    def smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Encourages smooth velocity transitions with geological realism."""
        # L1 smoothness (preserves edges better than L2)
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        # Weight by distance - closer points should be more similar
        h_weights = torch.exp(-0.1 * torch.arange(diff_h.shape[2], device=pred.device).float())
        w_weights = torch.exp(-0.1 * torch.arange(diff_w.shape[3], device=pred.device).float())
        
        weighted_h = torch.mean(diff_h * h_weights.view(1, 1, -1, 1))
        weighted_w = torch.mean(diff_w * w_weights.view(1, 1, 1, -1))
        
        return weighted_h + weighted_w
    
    def gradient_magnitude_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalizes unrealistic velocity gradients with geological constraints."""
        grad_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # Horizontal gradient
        grad_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # Vertical gradient (depth)
        
        # Calculate gradient magnitudes
        grad_magnitude = torch.sqrt(grad_h[:, :, :, :-1]**2 + grad_w[:, :, :-1, :]**2 + 1e-8)
        
        # Penalize extreme gradients more heavily (geological realism)
        # Typical seismic velocity gradients are < 2 km/s per 100m
        threshold = 0.5  # Reasonable gradient threshold
        penalty = torch.where(grad_magnitude > threshold, 
                            (grad_magnitude - threshold)**2, 
                            grad_magnitude**2 * 0.1)
        
        return torch.mean(penalty)
    
    
    def depth_consistency_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Enforce that velocity generally increases with depth (geological realism)."""
        # Velocity should generally increase with depth in many geological settings
        depth_gradient = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # Positive = increasing with depth
        
        # Encourage positive depth gradients but don't enforce strictly
        # (some geological formations can have velocity inversions)
        negative_gradient_penalty = torch.mean(F.relu(-depth_gradient - 0.1)**2)
        
        return negative_gradient_penalty
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Main prediction loss
        mape = self.mape_loss(prediction, target)
        
        # Physics-informed losses with adaptive weighting
        tv = self.tv_loss(prediction)
        smoothness = self.smoothness_loss(prediction)
        gradient = self.gradient_magnitude_loss(prediction)
        depth_consistency = self.depth_consistency_loss(prediction)
        
        # Apply adaptive weighting
        tv_weighted = self.get_adaptive_weight(self.tv_weight, self.current_epoch) * tv
        smoothness_weighted = self.get_adaptive_weight(self.smoothness_weight, self.current_epoch) * smoothness
        gradient_weighted = self.get_adaptive_weight(self.gradient_weight, self.current_epoch) * gradient
        depth_weighted = self.get_adaptive_weight(self.tv_weight * 0.5, self.current_epoch) * depth_consistency
        
        # Combined loss
        combined_loss = (mape + tv_weighted + smoothness_weighted + gradient_weighted + depth_weighted)
        
        return combined_loss, {
            "mape": mape,
            "tv": tv,
            "smoothness": smoothness,
            "gradient": gradient,
            "depth_consistency": depth_consistency,
            # Return weighted versions for monitoring
            "tv_weighted": tv_weighted,
            "smoothness_weighted": smoothness_weighted,
            "gradient_weighted": gradient_weighted,
            "depth_weighted": depth_weighted
        }

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

def cleanup_resources() -> None:
    """Clean up GPU memory and other resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory freed. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    gc.collect()
    logger.info("Python garbage collection completed")

def log_model_info(model: nn.Module) -> None:
    """Log model architecture information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    if torch.cuda.is_available():
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        logger.info(f"Model size: {model_size_mb:.1f} MB")

def setup_data_loaders(config: SimpleNamespace) -> Tuple[DataLoader, DataLoader]:
    """Setup training and validation data loaders with proper seeding."""
    logger.info("Loading datasets")
    full_dataset = SnsDataset(data_dir=config.train_data_dir, is_train=True)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Create deterministic train/val split using a fixed generator
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    logger.info(f"Split dataset: {train_size} train, {val_size} validation samples")
    
    # Create data loaders with proper seeding for reproducibility
    def seed_worker(_worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader

def setup_model_and_training(config: SimpleNamespace) -> Tuple[nn.Module, ModelEMA, torch.optim.Optimizer, nn.Module]:
    """Setup model, optimizer, and loss function."""
    logger.info("Initializing model")
    try:
        model = MultiTaskSnsCaformerModel(
            backbone_name=config.backbone,
            pretrained=config.pretrained,
            in_channels_stem=config.in_channels_stem,
            out_channels_stem=config.stem_out_channels,
            use_multi_task=config.use_multi_task,
            use_tta=config.use_tta
        ).to(config.device)
        
        log_model_info(model)
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        cleanup_resources()
        raise
    
    ema_model = ModelEMA(model, decay=config.ema_decay)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    logger.info(f"Initialized optimizer: lr={config.lr}, weight_decay={config.weight_decay}")
    
    # Enhanced loss function
    criterion = MultiTaskLoss(
        velocity_weight=1.0,
        reconstruction_weight=config.reconstruction_loss_weight,
        use_multi_task=config.use_multi_task,
        tv_weight=config.tv_loss_weight,
        smoothness_weight=config.smoothness_loss_weight,
        gradient_weight=config.gradient_loss_weight,
        use_adaptive_weights=config.use_adaptive_physics_weights,
        physics_warmup_epochs=config.physics_warmup_epochs,
        physics_max_weight_multiplier=config.physics_max_weight_multiplier,
        epochs=config.epochs
    )
    
    return model, ema_model, optimizer, criterion

def setup_scheduler_and_scaler(optimizer: torch.optim.Optimizer, config: SimpleNamespace, total_steps: int) -> Tuple[torch.optim.lr_scheduler._LRScheduler, torch.cuda.amp.GradScaler]:
    """Setup learning rate scheduler and gradient scaler."""
    scheduler = ConstantCosineLR(optimizer, total_steps=total_steps, pct_cosine=config.pct_cosine)
    scaler = torch.amp.GradScaler('cuda' if config.device.type == 'cuda' else 'cpu')
    return scheduler, scaler

def load_checkpoint_if_exists(model: nn.Module, ema_model: ModelEMA, optimizer: torch.optim.Optimizer, 
                            scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: torch.cuda.amp.GradScaler, 
                            config: SimpleNamespace) -> Tuple[int, float]:
    """Load checkpoint if it exists and return start_epoch and best_val_loss."""
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        try:
            logger.info(f"Resuming training from {config.resume_checkpoint}")
            checkpoint = torch.load(config.resume_checkpoint, map_location=config.device, weights_only=False)
            
            # Validate checkpoint has required keys
            required_keys = ['model_state_dict', 'ema_model_state_dict', 'optimizer_state_dict', 
                           'scheduler_state_dict', 'scaler_state_dict', 'epoch', 'best_val_loss']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError(f"Checkpoint missing required keys: {missing_keys}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.module.load_state_dict(checkpoint['ema_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Resumed from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {config.resume_checkpoint}: {e}")
            logger.info("Starting training from scratch")
            start_epoch = 0
            best_val_loss = float('inf')
    
    return start_epoch, best_val_loss

def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler, criterion: nn.Module, 
                   scaler: torch.cuda.amp.GradScaler, ema_model: ModelEMA, epoch: int, config: SimpleNamespace) -> None:
    """Train the model for one epoch."""
    model.train()
    
    # Update physics loss weights for current epoch
    if hasattr(criterion.velocity_loss, 'update_epoch'):
        criterion.velocity_loss.update_epoch(epoch)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda' if config.device.type == 'cuda' else 'cpu'):
            # Multi-task forward pass
            velocity_pred, seismic_recon = model(inputs, return_auxiliary=True)
            loss, loss_components = criterion(velocity_pred, seismic_recon, targets, inputs)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema_model.update(model)
        
        # Enhanced logging with more metrics
        if (batch_idx % config.logging_steps == 0) or (batch_idx == len(train_loader) - 1):
            # Log gradient norms for monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            
            logger.info(f"Epoch {epoch+1}/{config.epochs}, Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, MAPE={loss_components['mape'].item():.4f}, "
                      f"Recon={loss_components['reconstruction'].item():.4f}, "
                      f"LR={optimizer.param_groups[0]['lr']:.2e}, GradNorm={grad_norm:.4f}")
        
        pbar.set_postfix({
            'Total': f'{loss.item():.4f}',
            'MAPE': f'{loss_components["mape"].item():.4f}',
            'Recon': f'{loss_components["reconstruction"].item():.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

def validate_model(ema_model: ModelEMA, val_loader: DataLoader, criterion: nn.Module, 
                  epoch: int, config: SimpleNamespace) -> float:
    """Validate the model and return average validation loss."""
    ema_model.module.eval()
    val_losses = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            with torch.amp.autocast('cuda' if config.device.type == 'cuda' else 'cpu'):
                velocity_pred, seismic_recon = ema_model.module(inputs, return_auxiliary=True)
                loss, loss_components = criterion(velocity_pred, seismic_recon, targets, inputs)
                val_losses.append(loss_components["mape"].item())
    
    avg_val_loss = np.mean(val_losses)
    logger.info(f"Epoch {epoch+1}/{config.epochs}: Validation MAPE: {avg_val_loss:.4f}")
    
    # Log memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {memory_used:.1f}GB used, {memory_reserved:.1f}GB reserved")
    
    return avg_val_loss

def save_checkpoints(model: nn.Module, ema_model: ModelEMA, optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: torch.cuda.amp.GradScaler,
                    epoch: int, avg_val_loss: float, best_val_loss: float, config: SimpleNamespace) -> float:
    """Save model checkpoints and return updated best_val_loss."""
    # Save latest checkpoint
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config.latest_checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }, config.latest_checkpoint_path)
    except Exception as e:
        logger.warning(f"Failed to save latest checkpoint: {e}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        try:
            os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
            torch.save(ema_model.module.state_dict(), config.save_path)
            logger.info(f"New best model saved with MAPE: {best_val_loss:.4f}")
        except Exception as e:
            logger.warning(f"Failed to save best model: {e}")
    
    return best_val_loss

def run_enhanced_training() -> None:
    """Main training function with comprehensive logging and resource management."""
    logger.info("Starting enhanced training")
    
    # --- Setup ---
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)  # Ensure numpy operations are also seeded
    cleanup_resources()
    
    # --- Setup Components ---
    train_loader, val_loader = setup_data_loaders(cfg)
    model, ema_model, optimizer, criterion = setup_model_and_training(cfg)
    
    total_steps = cfg.epochs * len(train_loader)
    scheduler, scaler = setup_scheduler_and_scaler(optimizer, cfg, total_steps)
    
    start_epoch, best_val_loss = load_checkpoint_if_exists(model, ema_model, optimizer, scheduler, scaler, cfg)
    
    # --- Enhanced Training Loop ---
    for epoch in range(start_epoch, cfg.epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, ema_model, epoch, cfg)
        avg_val_loss = validate_model(ema_model, val_loader, criterion, epoch, cfg)
        best_val_loss = save_checkpoints(model, ema_model, optimizer, scheduler, scaler, epoch, avg_val_loss, best_val_loss, cfg)
        
        # Cleanup after each epoch
        if epoch % 10 == 0:  # Every 10 epochs
            cleanup_resources()
    
    logger.info("Training completed successfully")
    cleanup_resources()


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
    except Exception as e:
        print(f"Warning: Error loading existing submission file: {e}. Creating new file.")
        submission = dict({})
    
    submission.update({sample_id: prediction})
    
    try:
        os.makedirs(os.path.dirname(submission_path) if os.path.dirname(submission_path) else '.', exist_ok=True)
        np.savez(submission_path, **submission)
    except Exception as e:
        print(f"Error saving submission: {e}")
        raise

def run_enhanced_inference():
    """
    Enhanced inference function incorporating all model improvements including
    multi-task learning, progressive resolution, and robust prediction handling.
    """
    print("Starting enhanced inference...")
    
    # Initialize enhanced model with all improvements
    model = MultiTaskSnsCaformerModel(
        backbone_name=cfg.backbone,
        pretrained=False,
        in_channels_stem=cfg.in_channels_stem,
        out_channels_stem=cfg.stem_out_channels,
        use_multi_task=cfg.use_multi_task,
        use_tta=cfg.use_tta
    ).to(cfg.device)
    
    # Load trained weights
    try:
        checkpoint = torch.load(cfg.latest_checkpoint_path, map_location=cfg.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If it's just the model state dict directly
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model from {cfg.latest_checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {cfg.latest_checkpoint_path}")
        print("Please ensure you have trained the model first or update the checkpoint path")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Setup test dataset with enhanced preprocessing
    test_dataset = SnsDataset(data_dir=cfg.test_data_dir, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    submission_file = "enhanced_caformer_submission.npz"
    if os.path.exists(submission_file):
        os.remove(submission_file)
    
    print(f"Processing {len(test_dataset)} test samples...")
    
    # Enhanced inference with multiple prediction strategies
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(test_loader, desc="Enhanced Inference")):
            inputs = inputs.to(cfg.device)
            
            # 1. Standard prediction (already includes horizontal flip TTA)
            predictions = model(inputs)
            
            # Get sample IDs for this batch
            start_idx = i * cfg.batch_size
            end_idx = start_idx + len(inputs)
            batch_sample_ids = test_dataset.sample_ids[start_idx:end_idx]
            
            # Save predictions with enhanced quality checks
            for sample_id, pred_tensor in zip(batch_sample_ids, predictions):
                # Quality assurance checks
                pred_np = pred_tensor.squeeze(0).cpu().numpy()
                
                # Ensure physically reasonable velocity values
                pred_np = np.clip(pred_np, cfg.velocity_clip_min, cfg.velocity_clip_max)
                
                # Convert to required format
                prediction_np = pred_np.astype(np.float64)
                
                assert prediction_np.shape == (300, 1259), 'prediction shape mismatch'
                # Validation check
                bool, response = validate_prediction(prediction_np)
                if not bool:
                    print(f"Warning: Prediction quality check failed because of {response} for sample {sample_id}")
                
                create_submission(sample_id, prediction_np, submission_file)
    
    print(f"Enhanced inference complete. Submission file created at: {submission_file}")
    print(f"Final submission contains {len(test_dataset)} predictions")

def validate_prediction(prediction_np):
    """
    Perform quality validation on predictions.
    """
    # Check for NaN or Inf values
    if not np.isfinite(prediction_np).all():
        return False, "NaN or Inf values detected"
    
    # Check velocity range is reasonable
    if prediction_np.min() < 0.5 or prediction_np.max() > 10.0:
        return False, "Velocity out of expected range (0.5 to 10.0 km/s)"
    
    # Check for extreme gradients (unrealistic geological structures)
    grad_h = np.abs(np.diff(prediction_np, axis=0))
    grad_v = np.abs(np.diff(prediction_np, axis=1))
    
    if grad_h.max() > cfg.gradient_threshold or grad_v.max() > cfg.gradient_threshold:
        return False, "Extreme gradients detected"
    
    return True, "Good quality prediction"

# Modified main execution
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Enhanced SNS CAFormer")
    argument_parser.add_argument('--train', action='store_true', help='Run enhanced training')
    argument_parser.add_argument('--inference', action='store_true', help='Run enhanced inference')

    args = argument_parser.parse_args()
    
    if args.train:
        run_enhanced_training()
    elif args.inference:
        run_enhanced_inference()
    else:
        print("Please specify --train or --inference to run the respective mode.")