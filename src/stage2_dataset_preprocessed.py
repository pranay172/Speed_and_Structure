#!/usr/bin/env python3
"""
Simplified Stage2Dataset for preprocessed data.

This dataset class loads pre-computed Stage 2 training data, eliminating the need
for on-the-fly forward model computation and Stage 1 model inference during training.

The preprocessed data includes:
- combined_seismic.npy: Concatenated [original_seismic, deviation_seismic] (10, 10001, 31)
- stage1_velocity.npy: Stage 1 velocity prediction (300, 1259)
- residual_target.npy: Residual target δy = y_true - y1 (300, 1259)
"""

import os

# Repository root (parent of src/); override with the SNS_ROOT environment variable.
PROJECT_ROOT = os.environ.get("SNS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class PreprocessedStage2Dataset(Dataset):
    """
    Fast dataset for Stage 2 training using preprocessed data.
    
    This dataset simply loads pre-computed NumPy arrays, making training much faster
    and eliminating library conflicts from forward model computation.
    """
    
    def __init__(self, data_dir: str, is_train: bool = True):
        """
        Initialize dataset with preprocessed data directory.
        
        Args:
            data_dir: Directory containing preprocessed Stage 2 data
            is_train: Whether this is for training (enables augmentation)
        """
        self.data_dir = data_dir
        self.is_train = is_train
        
        # Get all sample directories
        self.sample_ids = [
            os.path.basename(p) for p in glob.glob(os.path.join(data_dir, '*')) 
            if os.path.isdir(p)
        ]
        
        if len(self.sample_ids) == 0:
            raise ValueError(f"No preprocessed samples found in directory: {data_dir}")
        
        # Validate a few samples to ensure preprocessing was successful
        self._validate_samples()
        
        print(f"✅ Loaded {len(self.sample_ids)} preprocessed Stage 2 samples from {data_dir}")

    def _validate_samples(self):
        """Validate that preprocessed samples have expected files and shapes."""
        expected_files = ['combined_seismic.npy', 'stage1_velocity.npy', 'residual_target.npy']
        
        # Check first few samples
        validation_samples = self.sample_ids[:min(3, len(self.sample_ids))]
        
        for sample_id in validation_samples:
            sample_path = os.path.join(self.data_dir, sample_id)
            
            # Check all expected files exist
            missing_files = []
            for file_name in expected_files:
                if not os.path.exists(os.path.join(sample_path, file_name)):
                    missing_files.append(file_name)
            
            if missing_files:
                raise ValueError(f"Sample {sample_id} missing files: {missing_files}")
            
            # Check shapes
            try:
                combined_seismic = np.load(os.path.join(sample_path, 'combined_seismic.npy'))
                stage1_velocity = np.load(os.path.join(sample_path, 'stage1_velocity.npy'))
                residual_target = np.load(os.path.join(sample_path, 'residual_target.npy'))
                
                # Validate shapes
                assert combined_seismic.shape == (10, 10001, 31), f"Combined seismic shape: {combined_seismic.shape}"
                assert stage1_velocity.shape == (300, 1259), f"Stage1 velocity shape: {stage1_velocity.shape}"
                assert residual_target.shape == (300, 1259), f"Residual target shape: {residual_target.shape}"
                
            except Exception as e:
                raise ValueError(f"Sample {sample_id} validation failed: {e}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load preprocessed data for a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (seismic_tensor, velocity_tensor, residual_tensor)
        """
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.data_dir, sample_id)
        
        # Load preprocessed data (much faster than computing on-the-fly!)
        combined_seismic = np.load(os.path.join(sample_path, 'combined_seismic.npy'))
        stage1_velocity = np.load(os.path.join(sample_path, 'stage1_velocity.npy'))
        
        # Convert to tensors
        seismic_tensor = torch.from_numpy(combined_seismic.astype(np.float32))
        velocity_tensor = torch.from_numpy(stage1_velocity.astype(np.float32)).unsqueeze(0)  # Add channel dim
        
        if self.is_train:
            # Load residual target for training
            residual_target = np.load(os.path.join(sample_path, 'residual_target.npy'))
            residual_tensor = torch.from_numpy(residual_target.astype(np.float32)).unsqueeze(0)  # Add channel dim
            
            # Data augmentation: horizontal flipping
            if torch.rand(1) < 1.0:  # 100% chance to flip for training
                # Flip seismic data (receivers and source order)
                seismic_tensor = torch.flip(seismic_tensor, dims=[2])  # Flip receivers
                seismic_tensor[:5] = torch.flip(seismic_tensor[:5], dims=[0])  # Flip source order (first 5 channels)
                seismic_tensor[5:] = torch.flip(seismic_tensor[5:], dims=[0])  # Flip deviation channels (last 5 channels)
                
                # Flip velocity and residual horizontally
                velocity_tensor = torch.flip(velocity_tensor, dims=[1])  # Flip width dimension
                residual_tensor = torch.flip(residual_tensor, dims=[1])   # Flip width dimension
            
            return seismic_tensor, velocity_tensor, residual_tensor
        else:
            # For test data, return empty residual target
            empty_residual = torch.empty(0)
            return seismic_tensor, velocity_tensor, empty_residual


# Example usage and testing
if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader
    
    # Test the preprocessed dataset
    print("🧪 Testing PreprocessedStage2Dataset...")
    
    data_dir = os.path.join(PROJECT_ROOT, "data/train_stage2_preprocessed")
    
    if not os.path.exists(data_dir):
        print(f"❌ Preprocessed data directory not found: {data_dir}")
        print("Run generate_stage1_predictions.py and preprocess_stage2_data_devito.py first.")
        exit(1)
    
    # Create dataset
    try:
        dataset = PreprocessedStage2Dataset(data_dir, is_train=True)
        print(f"✅ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        exit(1)
    
    # Test data loading speed
    print("⚡ Testing data loading speed...")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    start_time = time.time()
    
    # Load first few batches
    for i, (seismic_batch, velocity_batch, residual_batch) in enumerate(dataloader):
        if i >= 5:  # Test first 5 batches
            break
        
        print(f"Batch {i+1}:")
        print(f"  Seismic shape: {seismic_batch.shape}")
        print(f"  Velocity shape: {velocity_batch.shape}")
        print(f"  Residual shape: {residual_batch.shape}")
        
        # Check for NaN/inf values
        if torch.isnan(seismic_batch).any() or torch.isnan(velocity_batch).any() or torch.isnan(residual_batch).any():
            print("  ⚠️  Warning: NaN values detected!")
        
        if torch.isinf(seismic_batch).any() or torch.isinf(velocity_batch).any() or torch.isinf(residual_batch).any():
            print("  ⚠️  Warning: Inf values detected!")
    
    end_time = time.time()
    
    print(f"⏱️  Data loading time: {end_time - start_time:.2f} seconds for 5 batches")
    print(f"🚀 Average time per batch: {(end_time - start_time) / 5:.3f} seconds")
    
    print("\n✅ PreprocessedStage2Dataset test completed successfully!")
    print("🎉 Ready for fast Stage 2 training without forward model bottlenecks!")