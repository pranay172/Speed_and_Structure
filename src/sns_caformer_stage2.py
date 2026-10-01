#!/usr/bin/env python3
"""
SNS Competition Stage 2 Model - Residual Refinement Architecture

This implements the 2-stage approach from the 15th place Kaggle solution:
1. Stage 1: Predicts initial velocity field y1 from seismic data x
2. Stage 2: Refines y1 using residual learning with δx = x - Forward[y1] and δy = y_true - y1

The Stage 2 model takes:
- Input 1: Concatenated [original_seismic(5ch), deviation_seismic(5ch)] = 10 channels
- Input 2: Stage 1 velocity prediction y1
- Target: Velocity residual δy = y_true - y1

This allows iterative refinement where Stage 2 learns to correct Stage 1 errors.
"""

import os

# Repository root (parent of src/); override with the SNS_ROOT environment variable.
PROJECT_ROOT = os.environ.get("SNS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
from typing import Tuple, List, Dict
from types import SimpleNamespace

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

import timm

# Import forward model for generating simulated seismic data
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Stage 1 components
try:
    from sns_caformer_enhanced_notebook import (
        DecoderBlock, ModelEMA, ConstantCosineLR, setup_logging
    )
    STAGE1_COMPONENTS_AVAILABLE = True
    print("Successfully imported Stage 1 components")
except ImportError as e:
    print(f"Warning: Could not import Stage 1 components: {e}")
    print("You may need to run this from the same directory as sns_caformer_enhanced_notebook.py")
    STAGE1_COMPONENTS_AVAILABLE = False

# Configuration for Stage 2
cfg = SimpleNamespace()

# --- General ---
cfg.seed = 123
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
cfg.train_data_dir = os.path.join(PROJECT_ROOT, "data/train_stage2_preprocessed")  # Use preprocessed data
cfg.test_data_dir = os.path.join(PROJECT_ROOT, "data/test")

# --- Stage 1 Model Paths ---
cfg.stage1_model_path = os.path.join(PROJECT_ROOT, "models/caformer_sns_latest.pth")

# --- Stage 2 Model ---
cfg.backbone = "caformer_b36.sail_in22k_ft_in1k"
cfg.pretrained = True
cfg.in_channels_seismic = 10  # 5 original + 5 deviation channels
cfg.in_channels_velocity = 1  # Stage 1 velocity prediction
cfg.stem_out_channels = 128

# --- Training ---
cfg.epochs = 50
cfg.batch_size = 4  # Reduced due to additional inputs
cfg.lr = 1e-4  # LR for residual learning
cfg.weight_decay = 1e-5
cfg.ema_decay = 0.99

# --- Loss Weights (MAPE on final prediction) ---
cfg.mape_weight = 1.0

# --- Paths ---
cfg.save_path = os.path.join(PROJECT_ROOT, "models/caformer_sns_stage2_best.pth")
cfg.latest_checkpoint_path = os.path.join(PROJECT_ROOT, "models/caformer_sns_stage2_latest.pth")
cfg.resume_checkpoint = None

# --- Logging ---
cfg.logging_steps = 50
cfg.log_level = "INFO"
cfg.pct_cosine = 0.7

def validate_stage2_config(config: SimpleNamespace) -> None:
    """Validate Stage 2 specific configuration."""
    assert config.in_channels_seismic == 10, f"Stage 2 seismic input must be 10 channels, got {config.in_channels_seismic}"
    assert config.in_channels_velocity == 1, f"Stage 2 velocity input must be 1 channel, got {config.in_channels_velocity}"
    
    if not os.path.exists(config.stage1_model_path):
        warnings.warn(f"Stage 1 model not found at {config.stage1_model_path}")

validate_stage2_config(cfg)
logger = setup_logging(cfg)

def seed_worker(_worker_id: int) -> None:
    """Worker initialization function for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

class Stage2HybridStem(nn.Module):
    """
    Hybrid stem for Stage 2 that processes both seismic and velocity inputs.
    
    The seismic input (10 channels) and velocity input (1 channel) are processed
    separately and then combined via addition after appropriate convolutions.
    This version uses convolutional downsampling for the velocity stem to preserve
    more spatial information.
    """
    
    def __init__(self, in_channels_seismic: int = 10, in_channels_velocity: int = 1, 
                 out_channels: int = 128):
        super().__init__()
        
        # Seismic processing path (similar to Stage 1 but with 10 channels)
        self.seismic_stem = nn.Sequential(
            nn.Conv2d(in_channels_seismic, 32, kernel_size=(10, 3), stride=(10, 1), padding=(0, 1)),
            nn.InstanceNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(4, 1), padding=(0, 1)),
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)),
            nn.InstanceNorm2d(64), nn.GELU(),
            # Upsample to match target resolution
            nn.ConvTranspose2d(64, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(out_channels), nn.GELU()
        )
        
        # Velocity processing path with learned downsampling
        self.velocity_stem = nn.Sequential(
            # Layer 1: Initial convolution and downsampling
            nn.Conv2d(in_channels_velocity, 32, kernel_size=7, stride=(2, 4), padding=3), # H: 300->150, W: 1259->315
            nn.InstanceNorm2d(32), nn.GELU(),
            
            # Layer 2: Further downsampling and feature extraction
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # H: 150->150, W: 315->158
            nn.InstanceNorm2d(64), nn.GELU(),
            
            # Layer 3: Increase channels to match output
            nn.Conv2d(64, out_channels, kernel_size=3, stride=(1, 1), padding=1),
            nn.InstanceNorm2d(out_channels), nn.GELU(),
            
            # Final adaptive pool to precisely match the seismic stem's output size.
            # This is much less destructive as it operates on a smaller, feature-rich map.
            nn.AdaptiveAvgPool2d((125, 124))
        )

    def forward(self, seismic_input: torch.Tensor, velocity_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining seismic and velocity features.
        
        Args:
            seismic_input: Shape (B, 10, 10001, 31)
            velocity_input: Shape (B, 1, 300, 1259)
            
        Returns:
            Combined features: Shape (B, 128, ~125, ~124)
        """
        seismic_features = self.seismic_stem(seismic_input)
        velocity_features = self.velocity_stem(velocity_input)
        
        # Safeguard: Ensure spatial dimensions match before addition
        if seismic_features.shape[2:] != velocity_features.shape[2:]:
            velocity_features = F.interpolate(
                velocity_features, 
                size=seismic_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Combine via element-wise addition
        combined_features = seismic_features + velocity_features
        return combined_features


class Stage2Model(nn.Module):
    """
    Stage 2 Model for residual velocity refinement.
    
    Takes concatenated seismic signals and Stage 1 velocity prediction,
    outputs residual correction δy = y_true - y1.
    """
    
    def __init__(self, backbone_name: str, pretrained: bool = True, 
                 in_channels_seismic: int = 10, in_channels_velocity: int = 1,
                 out_channels_stem: int = 128, use_tta: bool = True):
        super().__init__()
        
        self.use_tta = use_tta
        
        # Hybrid stem for processing both inputs
        self.stem = Stage2HybridStem(
            in_channels_seismic=in_channels_seismic,
            in_channels_velocity=in_channels_velocity,
            out_channels=out_channels_stem
        )
        
        # Backbone (reuse from Stage 1)
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True, 
            in_chans=out_channels_stem
        )
        self.backbone.stem = nn.Identity()
        
        # Get encoder channels dynamically
        with torch.no_grad():
            dummy_seismic = torch.randn(2, in_channels_seismic, 10001, 31)
            dummy_velocity = torch.randn(2, in_channels_velocity, 300, 1259)
            stem_output = self.stem(dummy_seismic, dummy_velocity)
            features = self.backbone(stem_output)
            encoder_channels = [f.shape[1] for f in features]
        
        # Decoder for residual prediction (no multi-task learning needed)
        self.decoder = Stage2Decoder(encoder_channels=encoder_channels)

    def forward(self, seismic_input: torch.Tensor, velocity_input: torch.Tensor, 
                use_tta: bool = None) -> torch.Tensor:
        """
        Forward pass for Stage 2 model.
        
        Args:
            seismic_input: Combined seismic signals (B, 10, 10001, 31)
            velocity_input: Stage 1 velocity prediction (B, 1, 300, 1259)
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Residual velocity correction (B, 1, 300, 1259)
        """
        # Process inputs through hybrid stem
        stem_features = self.stem(seismic_input, velocity_input)
        
        # Extract features using backbone
        features = self.backbone(stem_features)
        
        # Decode to residual prediction
        residual_pred = self.decoder(features)
        
        # Apply TTA if specified and in evaluation mode
        if use_tta is None:
            use_tta = self.use_tta

        if not self.training and use_tta:
            residual_pred_tta = self._apply_tta(seismic_input, velocity_input)
            residual_pred = (residual_pred + residual_pred_tta) / 2
        
        return residual_pred
    
    def _apply_tta(self, seismic_input: torch.Tensor, velocity_input: torch.Tensor) -> torch.Tensor:
        """Apply test-time augmentation with horizontal flipping."""
        assert seismic_input.shape[1:] == (10, 10001, 31), f"Seismic input shape mismatch: {seismic_input.shape}"
        assert velocity_input.shape[1:] == (1, 300, 1259), f"Velocity input shape mismatch: {velocity_input.shape}"
        # Flip seismic data
        seismic_flipped = torch.flip(seismic_input, dims=[3])  # Flip receivers
        seismic_flipped[:, :5] = torch.flip(seismic_flipped[:, :5], dims=[1])  # Flip source order
        seismic_flipped[:, 5:] = torch.flip(seismic_flipped[:, 5:], dims=[1])  # Flip deviation channels

        # Flip velocity input
        velocity_flipped = torch.flip(velocity_input, dims=[2])  # Flip horizontally
        
        # Forward pass with flipped inputs
        stem_features = self.stem(seismic_flipped, velocity_flipped)
        features = self.backbone(stem_features)
        residual_pred_flipped = self.decoder(features)
        
        # Flip prediction back to original orientation
        residual_pred_unflipped = torch.flip(residual_pred_flipped, dims=[2])
        
        return residual_pred_unflipped


class Stage2Decoder(nn.Module):
    """Decoder for Stage 2 residual prediction."""
    
    def __init__(self, encoder_channels: List[int], decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)):
        super().__init__()
        
        encoder_channels = encoder_channels[::-1]  # Reverse for decoder
        head_channels = encoder_channels[0]
        
        # Decoder blocks
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        
        self.blocks = nn.ModuleList()
        for i_ch, s_ch, o_ch in zip(in_channels, skip_channels, decoder_channels):
            self.blocks.append(DecoderBlock(i_ch, s_ch, o_ch))
        
        # Residual prediction head
        self.residual_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

        # Initialize the final layer to predict zero.
        # This provides a strong starting point where the Stage 2 model initially
        # makes no correction, equaling the Stage 1 score.
        with torch.no_grad():
            self.residual_head.weight.zero_()
            self.residual_head.bias.zero_()
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]  # Reverse for decoder
        x = features[0]
        skips = features[1:] + [None]
        
        # Pass through decoder blocks
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
        
        # Generate residual prediction
        residual_pred = self.residual_head(x)
        
        # Resize to target shape (300, 1259)
        residual_pred = F.interpolate(
            residual_pred, 
            size=(1259, 300), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Transpose to match expected format
        residual_pred = torch.transpose(residual_pred, 2, 3)
        
        return residual_pred


def mape_loss(prediction, target):
    """Calculates the Mean Absolute Percentage Error with improved numerical stability."""
    epsilon = 1e-6  # Slightly larger epsilon for better stability
    target_safe = torch.clamp(target.abs(), min=epsilon)  # Avoid tiny denominators
    return torch.mean(torch.abs((target - prediction) / target_safe))

class Stage2Loss(nn.Module):
    """
    Loss function for Stage 2 that computes MAPE on the final reconstructed velocity.
    This directly optimizes the competition metric.
    """
    
    def __init__(self, mape_weight: float = 1.0):
        super().__init__()
        self.mape_weight = mape_weight
        self.loss_fn = mape_loss
        
    def forward(self, residual_pred: torch.Tensor, residual_target: torch.Tensor, stage1_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss based on the final reconstructed velocity with dual metrics.
        
        Args:
            residual_pred: Predicted residual δy_pred (B, 1, 300, 1259)
            residual_target: True residual δy_true (B, 1, 300, 1259)
            stage1_velocity: Stage 1 prediction y1 (B, 1, 300, 1259)
            
        Returns:
            Total loss and component breakdown
        """
        # Reconstruct the full velocity prediction
        final_pred = stage1_velocity + residual_pred
        
        # Reconstruct the full ground truth velocity
        final_target = stage1_velocity + residual_target
        
        # Calculate MAPE on the final reconstructed velocity (competition metric)
        mape_loss_val = self.loss_fn(final_pred, final_target)
        total_loss = self.mape_weight * mape_loss_val
        
        # Also track L1 loss on residual for monitoring residual learning quality
        residual_l1 = F.l1_loss(residual_pred, residual_target)
        
        return total_loss, {
            "mape": mape_loss_val,
            "residual_l1": residual_l1,
            "total": total_loss
        }


def setup_stage2_data_loaders(config: SimpleNamespace) -> Tuple[DataLoader, DataLoader]:
    """Setup Stage 2 training and validation data loaders using preprocessed data."""
    logger.info("Loading Stage 2 preprocessed datasets")
    
    # Import the preprocessed dataset class
    from stage2_dataset_preprocessed import PreprocessedStage2Dataset
    
    full_dataset = PreprocessedStage2Dataset(
        data_dir=config.train_data_dir,
        is_train=True
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    logger.info(f"Stage 2 dataset split: {train_size} train, {val_size} validation samples")
    
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
    
    logger.info(f"Created Stage 2 data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader


def setup_stage2_model_and_training(config: SimpleNamespace) -> Tuple[nn.Module, ModelEMA, torch.optim.Optimizer, nn.Module]:
    """Setup Stage 2 model, optimizer, and loss function."""
    logger.info("Initializing Stage 2 model")
    
    model = Stage2Model(
        backbone_name=config.backbone,
        pretrained=config.pretrained,
        in_channels_seismic=config.in_channels_seismic,
        in_channels_velocity=config.in_channels_velocity,
        out_channels_stem=config.stem_out_channels,
        use_tta=True
    ).to(config.device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Stage 2 model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    ema_model = ModelEMA(model, decay=config.ema_decay)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    criterion = Stage2Loss(mape_weight=config.mape_weight)
    
    return model, ema_model, optimizer, criterion


def train_stage2_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                          scheduler: torch.optim.lr_scheduler._LRScheduler, criterion: nn.Module,
                          scaler: torch.cuda.amp.GradScaler, ema_model: ModelEMA, epoch: int, config: SimpleNamespace) -> None:
    """Train Stage 2 model for one epoch."""
    model.train()
    
    pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{config.epochs} [Train]")
    
    for batch_idx, (seismic_inputs, velocity_inputs, residual_targets) in enumerate(pbar):
        seismic_inputs = seismic_inputs.to(config.device)
        velocity_inputs = velocity_inputs.to(config.device)
        residual_targets = residual_targets.to(config.device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda' if config.device.type == 'cuda' else 'cpu'):
            residual_pred = model(seismic_inputs, velocity_inputs, use_tta=False)
            loss, loss_components = criterion(residual_pred, residual_targets, velocity_inputs)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema_model.update(model)
        
        if (batch_idx % config.logging_steps == 0) or (batch_idx == len(train_loader) - 1):
            logger.info(f"Stage 2 Epoch {epoch+1}/{config.epochs}, Batch {batch_idx+1}/{len(train_loader)}: "
                       f"Loss={loss.item():.4f}, MAPE={loss_components['mape'].item():.4f}, "
                       f"Residual_L1={loss_components['residual_l1'].item():.4f}, "
                       f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'MAPE': f'{loss_components["mape"].item():.4f}',
            'ResL1': f'{loss_components["residual_l1"].item():.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })


def run_stage2_training() -> None:
    """Main Stage 2 training function."""
    logger.info("Starting Stage 2 training")
    
    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Check dependencies
    if not STAGE1_COMPONENTS_AVAILABLE:
        raise ImportError("Stage 1 components not available. Please ensure sns_caformer_enhanced_notebook.py is accessible.")
    
    # Setup components
    train_loader, val_loader = setup_stage2_data_loaders(cfg)
    model, ema_model, optimizer, criterion = setup_stage2_model_and_training(cfg)
    
    total_steps = cfg.epochs * len(train_loader)
    scheduler = ConstantCosineLR(optimizer, total_steps=total_steps, pct_cosine=cfg.pct_cosine)
    scaler = torch.amp.GradScaler('cuda' if cfg.device.type == 'cuda' else 'cpu')
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(cfg.epochs):
        train_stage2_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, ema_model, epoch, cfg)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seismic_inputs, velocity_inputs, residual_targets in tqdm(val_loader, desc=f"Stage 2 Epoch {epoch+1} [Val]"):
                seismic_inputs = seismic_inputs.to(cfg.device)
                velocity_inputs = velocity_inputs.to(cfg.device)
                residual_targets = residual_targets.to(cfg.device)
                
                with torch.amp.autocast('cuda' if cfg.device.type == 'cuda' else 'cpu'):
                    residual_pred = ema_model.module(seismic_inputs, velocity_inputs)
                    loss, _ = criterion(residual_pred, residual_targets, velocity_inputs)
                    val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"Stage 2 Epoch {epoch+1}/{cfg.epochs}: Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
            torch.save(ema_model.module.state_dict(), cfg.save_path)
            logger.info(f"New best Stage 2 model saved with loss: {best_val_loss:.4f}")
    
    logger.info("Stage 2 training completed successfully")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SNS Competition Stage 2 Model")
    parser.add_argument('--train', action='store_true', help='Run Stage 2 training')
    
    args = parser.parse_args()
    
    if args.train:
        run_stage2_training()
    else:
        print("Please specify --train to run Stage 2 training")
        print("Example: python sns_caformer_stage2.py --train")
